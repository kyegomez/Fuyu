from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor


main = FuyuImageProcessor()

pad = main.pad_image("agorabanner.jpg")
print(pad)
