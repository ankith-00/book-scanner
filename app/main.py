import io
import os
import uuid
import base64

import cv2
import numpy as np
import img2pdf
from flask import Flask, render_template, request, session, send_file, send_from_directory, redirect, url_for

app = Flask(__name__)
app.secret_key = 'super_secret_scanner_key'

A4_POINTS = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
A4_W_PX = 1654
A4_H_PX = 2339
A4_MARGIN_PX = 24
JPEG_QUALITY = 95
MAX_COVER_CROP_RATIO = 0.30
ROTATE_CLOCKWISE = True

SESSION_STORAGE = {}


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(app.root_path + '/images', filename)


@app.route('/', methods=['GET'])
def index():
    if 'batch_id' not in session:
        session['batch_id'] = uuid.uuid4().hex
        SESSION_STORAGE[session['batch_id']] = []

    pages = SESSION_STORAGE.get(session['batch_id'], [])
    encoded_pages = [
        base64.b64encode(finalize_page(p['data']) if p['kind'] == 'mobile' else p['data']).decode('utf-8')
        for p in pages
    ]
    return render_template('index.html', pages=encoded_pages)


def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_page(img):
    ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return encoded.tobytes() if ok else None


def split_spread(image_bytes, split_ratio=0.5, crop_top=0.0, crop_bottom=1.0, crop_left=0.0, crop_right=1.0, skip_split=False):
    img = decode_image(image_bytes)
    if img is None:
        return [], 'desktop'

    height, width = img.shape[:2]
    axis = 0 if height >= width else 1

    y0 = int(round(height * crop_top))
    y1 = int(round(height * crop_bottom))
    x0 = int(round(width * crop_left))
    x1 = int(round(width * crop_right))
    img = img[y0:y1, x0:x1]
    height, width = img.shape[:2]

    if skip_split:
        parts = [img]
    else:
        if axis == 0:
            total, start, span = height, crop_top, crop_bottom - crop_top
        else:
            total, start, span = width, crop_left, crop_right - crop_left
        ratio = (split_ratio - start) / max(span, 1e-6)
        ratio = max(0.02, min(0.98, ratio))
        mid = max(1, int(round(total * ratio)))

        if axis == 1:
            parts = [img[:, :mid], img[:, mid:]]
        else:
            parts = [img[:mid, :], img[mid:, :]]

    raw_buffers = []
    for part in parts:
        if part.size == 0:
            continue
        if axis == 0:
            part = cv2.rotate(part, cv2.ROTATE_90_COUNTERCLOCKWISE if ROTATE_CLOCKWISE else cv2.ROTATE_90_CLOCKWISE)
        buffer = encode_page(part)
        if buffer:
            raw_buffers.append(buffer)
    return raw_buffers, ('mobile' if axis == 0 else 'desktop')


def place_on_a4(page):
    page_h, page_w = page.shape[:2]
    a4_ar = A4_W_PX / A4_H_PX
    page_ar = page_w / page_h

    cover_overflow = max(page_ar / a4_ar, a4_ar / page_ar) - 1.0
    if cover_overflow <= MAX_COVER_CROP_RATIO:
        scale = max(A4_W_PX / page_w, A4_H_PX / page_h)
        target = (max(A4_W_PX + 4, int(np.ceil(page_w * scale))),
                  max(A4_H_PX + 4, int(np.ceil(page_h * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        page = cv2.resize(page, target, interpolation=interp)

        shown_h, shown_w = page.shape[:2]
        x0 = max(0, (shown_w - A4_W_PX) // 2)
        y0 = max(0, (shown_h - A4_H_PX) // 2)
        return page[y0:y0 + A4_H_PX, x0:x0 + A4_W_PX]

    canvas = np.full((A4_H_PX, A4_W_PX, 3), 255, dtype=np.uint8)
    max_w = A4_W_PX - 2 * A4_MARGIN_PX
    max_h = A4_H_PX - 2 * A4_MARGIN_PX
    scale = min(max_w / page_w, max_h / page_h)
    target = (max(1, int(round(page_w * scale))), max(1, int(round(page_h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    page = cv2.resize(page, target, interpolation=interp)
    placed_h, placed_w = page.shape[:2]
    x = (A4_W_PX - placed_w) // 2
    y = (A4_H_PX - placed_h) // 2
    canvas[y:y + placed_h, x:x + placed_w] = page
    return canvas


def finalize_page(raw_page_buffer):
    page = decode_image(raw_page_buffer)
    if page is None:
        return None
    page = place_on_a4(page)
    return encode_page(page)


@app.route('/capture', methods=['POST'])
def capture():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    if file.filename == '':
        return {"error": "Empty filename"}, 400

    batch_id = session.get('batch_id')
    if not batch_id:
        batch_id = uuid.uuid4().hex
        session['batch_id'] = batch_id

    if batch_id not in SESSION_STORAGE:
        SESSION_STORAGE[batch_id] = []

    image_bytes = file.read()

    try:
        split_ratio = float(request.form.get('split_ratio') or 0.5)
    except (TypeError, ValueError):
        split_ratio = 0.5
    split_ratio = max(0.02, min(0.98, split_ratio))

    try:
        crop_top = float(request.form.get('crop_top') or 0.0)
        crop_bottom = float(request.form.get('crop_bottom') or 1.0)
    except (TypeError, ValueError):
        crop_top, crop_bottom = 0.0, 1.0
    crop_top = max(0.0, min(1.0, crop_top))
    crop_bottom = max(0.0, min(1.0, crop_bottom))
    if crop_bottom - crop_top < 0.05:
        crop_top, crop_bottom = 0.0, 1.0

    try:
        crop_left = float(request.form.get('crop_left') or 0.0)
        crop_right = float(request.form.get('crop_right') or 1.0)
    except (TypeError, ValueError):
        crop_left, crop_right = 0.0, 1.0
    crop_left = max(0.0, min(1.0, crop_left))
    crop_right = max(0.0, min(1.0, crop_right))
    if crop_right - crop_left < 0.05:
        crop_left, crop_right = 0.0, 1.0

    skip_split = (request.form.get('skip_split') or '').strip() in ('1', 'true', 'yes')
    new_raw_halves, kind = split_spread(
        image_bytes,
        split_ratio=split_ratio,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        skip_split=skip_split
    )
    SESSION_STORAGE[batch_id].extend([{'kind': kind, 'data': raw} for raw in new_raw_halves])

    new_encoded = []
    for raw in new_raw_halves:
        preview = finalize_page(raw) if kind == 'mobile' else raw
        if not preview:
            preview = raw
        new_encoded.append(base64.b64encode(preview).decode('utf-8'))

    return {
        'status': 'success',
        'total_pages': len(SESSION_STORAGE[batch_id]),
        'new_pages': new_encoded
    }


@app.route('/clear', methods=['POST'])
def clear_session():
    batch_id = session.get('batch_id')
    if batch_id in SESSION_STORAGE:
        SESSION_STORAGE[batch_id] = []
    return redirect(url_for('index'))


@app.route('/export-pdf')
def export_pdf():
    pdf_bytes = build_pdf()
    if pdf_bytes is None:
        return "No pages scanned", 400

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name="scanned_book.pdf"
    )


def build_pdf():
    batch_id = session.get('batch_id')
    stored_pages = SESSION_STORAGE.get(batch_id, [])
    if not stored_pages:
        return None

    pdf_pages = []
    for entry in stored_pages:
        if entry['kind'] == 'mobile':
            page = finalize_page(entry['data'])
        else:
            page = entry['data']
        if page:
            pdf_pages.append(page)

    if not pdf_pages:
        return None

    return img2pdf.convert(pdf_pages, layout_fun=img2pdf.get_layout_fun(A4_POINTS))


def main():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':
    main()
