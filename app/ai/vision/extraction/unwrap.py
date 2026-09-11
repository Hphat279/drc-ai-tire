import cv2
import numpy as np


def unwrap_tire(img: np.ndarray) -> np.ndarray | None:
    """
    Chuyển đổi bề mặt lốp xe dạng tròn hoặc elip
    thành dạng biểu diễn hình chữ nhật đã được trải phẳng.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, tire_mask = cv2.threshold(
        gray,
        5,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        tire_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    if len(largest_contour) >= 5:
        (xc, yc), (d1, d2), ellipse_angle = (
            cv2.fitEllipse(largest_contour)
        )
    else:
        (xc, yc), radius = cv2.minEnclosingCircle(
            largest_contour
        )

        d1 = radius * 2
        d2 = radius * 2
        ellipse_angle = 0

    r_out = int(max(d1, d2) / 2)

    ys, xs = np.where(tire_mask > 0)

    dists = np.sqrt(
        (xs - xc) ** 2 +
        (ys - yc) ** 2
    )

    r_in = max(
        int(np.percentile(dists, 8) * 0.82),
        int(r_out * 0.45),
    )

    out_h = r_out - r_in
    out_w = 2268

    theta_raw = np.linspace(
        0,
        2 * np.pi,
        out_w,
        dtype=np.float32,
    )

    aspect = d1 / d2

    theta = np.arctan2(
        np.sin(theta_raw),
        np.cos(theta_raw) * aspect,
    )

    theta += np.radians(ellipse_angle)

    r_arr = np.linspace(
        r_out,
        r_in,
        out_h,
        dtype=np.float32,
    )

    theta_grid, radius_grid = np.meshgrid(
        theta,
        r_arr,
    )

    map_x = np.clip(
        xc + radius_grid * np.cos(theta_grid),
        0,
        img.shape[1] - 1,
    ).astype(np.float32)

    map_y = np.clip(
        yc + radius_grid * np.sin(theta_grid),
        0,
        img.shape[0] - 1,
    ).astype(np.float32)

    img_unwrapped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    h = img_unwrapped.shape[0]

    img_unwrapped = img_unwrapped[
        :int(h * 0.98),
        :,
    ]

    img_unwrapped = cv2.resize(
        img_unwrapped,
        (
            img_unwrapped.shape[1] * 2,
            img_unwrapped.shape[0] * 2,
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    return img_unwrapped