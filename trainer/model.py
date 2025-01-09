import transformers
from models.codebook import CodebookAdapterModel, CodebookConfig


def load_codebook_model(model, codebook_size, inserted_layer, select_len):
    codebook_config = CodebookConfig(  
        codebook_size=codebook_size,
        inserted_layer=inserted_layer,
        select_len=select_len,
    )

    model = CodebookAdapterModel(model, codebook_config)
    return model
