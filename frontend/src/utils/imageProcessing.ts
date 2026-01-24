/**
 * Point interface representing x and y coordinates
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Calculates the perspective transform (homography) matrix that maps points from src to dst.
 * Returns a 3x3 matrix as a 9-element array.
 *
 * Solves the system of linear equations:
 * x' = (h0*x + h1*y + h2) / (h6*x + h7*y + h8)
 * y' = (h3*x + h4*y + h5) / (h6*x + h7*y + h8)
 *
 * We assume h8 = 1.
 */
function getPerspectiveTransform(src: Point[], dst: Point[]): number[] {
  const a: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const x = src[i].x;
    const y = src[i].y;
    const xp = dst[i].x;
    const yp = dst[i].y;

    a.push([x, y, 1, 0, 0, 0, -x * xp, -y * xp]);
    a.push([0, 0, 0, x, y, 1, -x * yp, -y * yp]);
    b.push(xp);
    b.push(yp);
  }

  // Gaussian elimination
  const n = 8;
  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(a[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(a[k][i]) > maxEl) {
        maxEl = Math.abs(a[k][i]);
        maxRow = k;
      }
    }

    for (let k = i; k < n; k++) {
      const tmp = a[maxRow][k];
      a[maxRow][k] = a[i][k];
      a[i][k] = tmp;
    }
    const tmp = b[maxRow];
    b[maxRow] = b[i];
    b[i] = tmp;

    for (let k = i + 1; k < n; k++) {
      const c = -a[k][i] / a[i][i];
      for (let j = i; j < n; j++) {
        if (i === j) {
          a[k][j] = 0;
        } else {
          a[k][j] += c * a[i][j];
        }
      }
      b[k] += c * b[i];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += a[i][j] * x[j];
    }
    x[i] = (b[i] - sum) / a[i][i];
  }

  return [...x, 1];
}

/**
 * Warps an image using the provided 4 source points to a rectangular destination.
 *
 * @param imageSource The source image (HTMLImageElement)
 * @param srcPoints Array of 4 points [TL, TR, BR, BL] in the source image coordinates
 * @returns A Promise that resolves to the cropped Blob
 */
export async function warpImage(
  imageSource: HTMLImageElement,
  srcPoints: Point[]
): Promise<Blob> {
  // 1. Calculate destination dimensions
  // Width = max(distance(TL, TR), distance(BL, BR))
  // Height = max(distance(TL, BL), distance(TR, BR))

  const distance = (p1: Point, p2: Point) =>
    Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

  const widthTop = distance(srcPoints[0], srcPoints[1]);
  const widthBottom = distance(srcPoints[3], srcPoints[2]);
  const width = Math.max(widthTop, widthBottom);

  const heightLeft = distance(srcPoints[0], srcPoints[3]);
  const heightRight = distance(srcPoints[1], srcPoints[2]);
  const height = Math.max(heightLeft, heightRight);

  const dstPoints: Point[] = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];

  // 2. Get Homography Matrix (Dst -> Src)
  // We want to iterate over destination pixels and find their corresponding source pixel
  // So we need the transform from Dst to Src.
  // Usually getPerspectiveTransform calculates Src -> Dst.
  // So we calculate Src -> Dst then invert it, OR just calculate Dst -> Src directly.
  // Let's calculate Dst -> Src directly to avoid inversion issues if possible,
  // but standard is Src -> Dst then invert for backward mapping.
  // Let's stick to calculating Dst -> Src directly:
  // Mapping from (u,v) in dest to (x,y) in src.
  const h = getPerspectiveTransform(dstPoints, srcPoints);

  // 3. Create Canvas and process
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get 2d context");

  // Draw original image to a temp canvas to get pixel data
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = imageSource.naturalWidth;
  tempCanvas.height = imageSource.naturalHeight;
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) throw new Error("Could not get temp 2d context");
  tempCtx.drawImage(imageSource, 0, 0);

  const srcData = tempCtx.getImageData(
    0,
    0,
    tempCanvas.width,
    tempCanvas.height
  );
  const dstData = ctx.createImageData(width, height);

  const srcPixels = srcData.data;
  const dstPixels = dstData.data;
  const srcWidth = tempCanvas.width;
  const srcHeight = tempCanvas.height;

  // 4. Pixel Loop (Bilinear Interpolation)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Apply homography to find source coordinates
      const d = h[6] * x + h[7] * y + h[8];
      const srcX = (h[0] * x + h[1] * y + h[2]) / d;
      const srcY = (h[3] * x + h[4] * y + h[5]) / d;

      // Bilinear Interpolation
      if (
        srcX >= 0 &&
        srcX < srcWidth - 1 &&
        srcY >= 0 &&
        srcY < srcHeight - 1
      ) {
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = x0 + 1;
        const y1 = y0 + 1;

        const dx = srcX - x0;
        const dy = srcY - y0;

        const i00 = (y0 * srcWidth + x0) * 4;
        const i10 = (y0 * srcWidth + x1) * 4;
        const i01 = (y1 * srcWidth + x0) * 4;
        const i11 = (y1 * srcWidth + x1) * 4;

        const dstIdx = (y * width + x) * 4;

        for (let c = 0; c < 3; c++) {
          // RGB
          const val =
            srcPixels[i00 + c] * (1 - dx) * (1 - dy) +
            srcPixels[i10 + c] * dx * (1 - dy) +
            srcPixels[i01 + c] * (1 - dx) * dy +
            srcPixels[i11 + c] * dx * dy;
          dstPixels[dstIdx + c] = val;
        }
        dstPixels[dstIdx + 3] = 255; // Alpha
      }
    }
  }

  ctx.putImageData(dstData, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Failed to create blob"));
      },
      "image/jpeg",
      0.95
    );
  });
}
