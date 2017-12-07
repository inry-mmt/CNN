import os
import re
import shutil
from sorter_inceptionV3 import Sorter

def main():
    classes = ["tumor", "non-tumor"]
    BASE_DIR = "/media/bioinfo/fatdata/generated_tiles"

    BASE_DIR_4_SAVE = "/media/bioinfo/fatdata/tumor_tiles"

    dirnames = sorted([name for name in os.listdir(BASE_DIR) if not re.match(r"\..+", name)])
    eliminatenames = [name for name in os.listdir("/media/bioinfo/fatdata/tumor_tiles") if not re.match(r"\..+", name)]

    for el in eliminatenames:
        if el in dirnames:
            dirnames.remove(el)

    sorter = Sorter(
        classes=classes,
        finetuning_weights_path="./tumor-non_tumor-ver1.h5",
        img_size=(300, 300),
    )

    regx = re.compile(r"\..+")
    for dirname in dirnames:
        tilenames = [name for name in os.listdir(os.path.join(BASE_DIR, dirname)) if not regx.match(name)]

        save_dir = os.path.join(BASE_DIR_4_SAVE, dirname)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print("picking out from {}".format(save_dir))
        counter = 0
        n_tiles = len(tilenames)

        for i, tilename in enumerate(tilenames):
            tile_path = os.path.join(BASE_DIR, dirname, tilename)

            evaluated_classname = sorter.detect(tile_path)
            if evaluated_classname == "tumor":
                out_path = os.path.join(save_dir, tilename)
                shutil.copyfile(tile_path, out_path)
                counter += 1
            print("{0}/{1}. {2} tiles is regarded as a tumor tile".format(i + 1, n_tiles, counter), "\r", end="")

        print("")
        print("{} tiles were tumor. They picked out and save at {}".format(counter, save_dir))

if __name__ == "__main__":
    main()