import datasets
import os

_CITATION = ""
_DESCRIPTION = "" 
_URL = ""
_HOMEPAGE = ""
_LICENSE = ""

class Spacenet(datasets.GeneratorBasedBuilder):
  
    DEFAULT_WRITER_BATCH_SIZE = 1
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="clean", description="Train Set.")]
  
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
            {
                "image": datasets.Image(),
                "text": datasets.Value("string"),
                "conditioning_image": datasets.Image()
            }
            )
        )
  
    def _split_generators(self, dl_manager):
        data_dir = dl_manager.extract(self.config.data_dir)
        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                name="train", gen_kwargs={"files": data_dir}
                )
            ]
        else:
            train_splits = [
                datasets.SplitGenerator(
                name="train", gen_kwargs={"files": data_dir}
                )
            ]
        return train_splits
  
    
    def _generate_examples(self, files):
        # DATA_DIR = "F:/Shahmir/ControlNet Satellite Imagery/diffusers/examples/controlnet/data.zip" 
        key = 0
        examples = list()
        image_dir = os.path.join(files, "images")
        condition_dir = os.path.join(files, "conditions")
        image_files = os.listdir(image_dir)
        condition_files = os.listdir(condition_dir)


        for i in range(len(image_files)):
            res = dict()
            res["image"] = "{}".format(os.path.join(image_dir, image_files[i]))
            res["text"] = ""
            res["conditioning_image"] = "{}".format(os.path.join(condition_dir, condition_files[i]))
            examples.append(res)
        
        for example in examples:
            yield key, {**example}
            key += 1
            examples = list()