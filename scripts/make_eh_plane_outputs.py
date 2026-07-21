import csv
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "hfss_outputs" / "fullarray_broadside" / "fullarray_gain_total_theta_phi.csv"
OUT = ROOT / "hfss_outputs" / "eh_planes"
OUT.mkdir(parents=True, exist_ok=True)


def read_hfss_long_table(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "theta": float(row["Theta[deg]"]),
                    "phi": float(row["Phi[deg]"]),
                    "gain": float(row["dB(GainTotal)"]),
                }
            )
    return rows


def write_clean_csv(path, plane, phi, rows, peak_gain):
    cut = sorted((r for r in rows if r["phi"] == phi), key=lambda r: r["theta"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plane", "Theta_deg", "Phi_deg", "GainTotal_dBi", "GainTotal_relative_dB"])
        for r in cut:
            writer.writerow([plane, f"{r['theta']:.0f}", f"{r['phi']:.0f}", f"{r['gain']:.6f}", f"{r['gain'] - peak_gain:.6f}"])
    return cut


def png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def save_png(path, width, height, rgb_rows):
    raw = b"".join(bytes([0]) + row for row in rgb_rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw, 9))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def draw_line(img, width, height, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                if 0 <= xx < width and 0 <= yy < height:
                    img[yy][xx] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def make_line_png(path, e_cut, h_cut, peak_gain):
    width, height = 1100, 620
    left, right, top, bottom = 90, 1040, 70, 520
    img = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    axis = (45, 45, 45)
    grid = (220, 220, 220)
    red = (214, 55, 55)
    blue = (45, 105, 195)

    def x_for(theta):
        return int(left + theta / 90.0 * (right - left))

    def y_for(rel_db):
        rel_db = max(-40.0, min(0.0, rel_db))
        return int(top + (0.0 - rel_db) / 40.0 * (bottom - top))

    for db in range(-40, 1, 10):
        y = y_for(db)
        draw_line(img, width, height, left, y, right, y, grid)
    for theta in range(0, 91, 15):
        x = x_for(theta)
        draw_line(img, width, height, x, top, x, bottom, grid)
    draw_line(img, width, height, left, top, left, bottom, axis)
    draw_line(img, width, height, left, bottom, right, bottom, axis)
    draw_line(img, width, height, right, top, right, bottom, axis)
    draw_line(img, width, height, left, top, right, top, axis)

    for cut, color in ((e_cut, red), (h_cut, blue)):
        pts = [(x_for(r["theta"]), y_for(r["gain"] - peak_gain)) for r in cut]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw_line(img, width, height, x0, y0, x1, y1, color)

    # Simple color legend bars.
    for y in range(44, 56):
        for x in range(735, 775):
            img[y][x] = red
        for x in range(835, 875):
            img[y][x] = blue

    save_png(path, width, height, [bytes(v for pixel in row for v in pixel) for row in img])


def main():
    rows = read_hfss_long_table(SRC)
    peak = max(r["gain"] for r in rows)
    e_cut = write_clean_csv(OUT / "e_plane_phi0_gain_total.csv", "E", 0.0, rows, peak)
    h_cut = write_clean_csv(OUT / "h_plane_phi90_gain_total.csv", "H", 90.0, rows, peak)

    with (OUT / "eh_plane_gain_total_clean.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plane", "Theta_deg", "Phi_deg", "GainTotal_dBi", "GainTotal_relative_dB"])
        for plane, cut in (("E", e_cut), ("H", h_cut)):
            for r in cut:
                writer.writerow([plane, f"{r['theta']:.0f}", f"{r['phi']:.0f}", f"{r['gain']:.6f}", f"{r['gain'] - peak:.6f}"])

    make_line_png(OUT / "eh_plane_1d_gain_total.png", e_cut, h_cut, peak)

    for plane, cut in (("E", e_cut), ("H", h_cut)):
        cut_peak = max(cut, key=lambda r: r["gain"])
        first_3db = next((r for r in cut if r["gain"] <= peak - 3.0), None)
        first_3db_theta = f"{first_3db['theta']:.0f}" if first_3db else "NA"
        print(
            f"{plane}-plane phi={cut_peak['phi']:.0f} peak theta={cut_peak['theta']:.0f} "
            f"gain={cut_peak['gain']:.4f} dBi first<=-3dB theta="
            f"{first_3db_theta}"
        )


if __name__ == "__main__":
    main()
