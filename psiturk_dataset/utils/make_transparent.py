import argparse
import cv2


def make_transparent(path, output_path):
    src = cv2.imread(path)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite(output_path, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default=""
    )
    parser.add_argument(
        "--output-path", type=str, default=""
    )
    args = parser.parse_args()
    make_transparent(args.path, args.output_path)
