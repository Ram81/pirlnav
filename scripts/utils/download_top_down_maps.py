import requests


scenes = [
    '00848-ziup5kvtCCR', '00839-zt1RVoi7PcG', '00820-mL8ThkuaVTM', '00814-p53SfW6mjZe',
    '00890-6s7QHgap2fW', '00853-5cdEh9F2hJL', '00880-Nfvxx8J5NCo', '00802-wcojb4TFT35',
    '00800-TEEsavR23oF', '00829-QaLdnwvtxbs', '00832-qyAac8rV8Zk', '00824-Dd4bFSTQ8gi',
    '00876-mv2HUxq3B53', '00878-XB4GS9ShBRE', '00877-4ok3usBNeis', '00835-q3zU7Yy5E5s',
    '00873-bxsVRursffK', '00813-svBbv1Pavdk', '00843-DYehNKdT76V', '00891-cvZr5TUy5C5'
]

single_floor_scenes = [
    "00813-svBbv1Pavdk", "00824-Dd4bFSTQ8gi", "00829-QaLdnwvtxbs",
    "00848-ziup5kvtCCR", "00853-5cdEh9F2hJL", "00876-mv2HUxq3B53",
    "00880-Nfvxx8J5NCo"
]


def download_images():
    download_url = "https://habitatwebsite.s3.amazonaws.com/website-visualization/{}/topdown_floors.png"
    for scene in scenes:
        url = download_url.format(scene)
        img_data = requests.get(url).content

        with open("demos/original/{}.png".format(scene), "wb") as handler:
            handler.write(img_data)


if __name__ == "__main__":
    download_images()
