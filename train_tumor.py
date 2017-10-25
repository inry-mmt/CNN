from sorter_inceptionV3 import Sorter

def main():
    classes = ["tumor", "non-tumor"]
    train_dir = "/home/bioinfo/ml/data/hyper_mutation/source/tumor_non-tumor/train"
    validation_dir = "/home/bioinfo/ml/data/hyper_mutation/source/tumor_non-tumor/validation"

    #classes = ['empty', 'notempty']
    #train_dir = "/home/bioinfo/share/aorb/train"
    #validation_dir = "/home/bioinfo/share/aorb/test"

    sorter = Sorter(
        classes=classes,
        train_dir=train_dir,
        validation_dir=validation_dir,
        save_weights_path="./tumor_non-tumor_randomize4.h5",
        img_size=(300, 300),
    )

    # train
    sorter.train()

if __name__ == "__main__":
    main()