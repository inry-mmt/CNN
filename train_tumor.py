from sorter_inceptionV3 import Sorter

def main():
    classes = ["tumor", "non-tumor"]
    train_dir = "/home/bioinfo/ml/data/hyper_mutation/source/tumor_non-tumor_more_trainset/train"
    validation_dir = "/home/bioinfo/ml/data/hyper_mutation/source/tumor_non-tumor_more_trainset/validation"

    #classes = ['empty', 'notempty']
    #train_dir = "/home/bioinfo/share/aorb/train"
    #validation_dir = "/home/bioinfo/share/aorb/test"

    sorter = Sorter(classes=classes, train_dir=train_dir, validation_dir=validation_dir)

    # train
    sorter.train()

if __name__ == "__main__":
    main()