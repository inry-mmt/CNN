from sorter_inceptionV3 import Sorter

def main():
    classes = ["hyper", "non-hyper"]
    train_dir = "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper/train"
    validation_dir = "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper/validation"

    #classes = ['empty', 'notempty']
    #train_dir = "/home/bioinfo/share/aorb/train"
    #validation_dir = "/home/bioinfo/share/aorb/test"

    sorter = Sorter(
        classes=classes,
        train_dir=train_dir,
        validation_dir=validation_dir,
        weights_path="./hyper_non-hyper.h5"
    )

    # train
    sorter.train()

if __name__ == "__main__":
    main()