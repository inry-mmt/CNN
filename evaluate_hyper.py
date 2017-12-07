from sorter_inceptionV3 import Sorter

def main():
    classes = ["hyper", "non-hyper"]
    validation_dirs = [
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/G-18-13",
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/2014-07859-12",
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/2015-01937-3-4",
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/G-12-2",
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/G-13-3",
        "/home/bioinfo/ml/data/hyper_mutation/source/hyper_non-hyper_171031.1129/kobetsu/NC-26",
        #"/media/bioinfo/fatdata/tumor_tiles_zvalue/auto(evaluation_format)/10-6259-30"
    ]

    #classes = ['empty', 'notempty']
    #train_dir = "/home/bioinfo/share/aorb/train"
    #validation_dir = "/home/bioinfo/share/aorb/test"

    for validation_dir in validation_dirs:
        print("\n\n\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n\n")
        print(validation_dir)
        sorter = Sorter(
            classes=classes,
            validation_dir=validation_dir,
            finetuning_weights_path="./hyper_non-hyper_many_zvalued_1.h5",
            img_size=(300, 300),
        )

        # train
        sorter.evaluate()

if __name__ == "__main__":
    main()