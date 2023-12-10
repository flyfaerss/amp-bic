# DATASET

## Visual Genome
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code we use to generate them is located at ```datasets/vg/generate_attribute_labels.py```. Although, we encourage later researchers to explore the value of attribute features, in our paper "Unbiased Scene Graph Generation from Biased Training", we follow the conventional setting to turn off the attribute head in both detector pretraining part and relationship prediction part for fair comparison, so does the default setting of this codebase.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `pysgg/config/paths_catelog.py`. 
2. Download the [scene graphs](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779871&authkey=AA33n7BRpB1xa3I) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `pysgg/config/paths_catelog.py`.
3. Link the image into the project folder
```
ln -s /path-to-vg/VG_100K datasets/vg/stanford_spilt/VG_100k_images
ln -s /path-to-vg/VG-SGG-with-attri.h5 datasets/vg/stanford_spilt/VG-SGG-with-attri.h5
```

## Openimage V4/V6 

### Download
The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).
The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 
You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3),
[Openimage V4(28GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EVWy0xJRx8RNo-zHF5bdANMBTYt6NvAaA59U32o426bRqw?e=6ygqFR)
The dataset dir contains the `images` and `annotations` folder. Link the `open_image_v4` and `open_image_v6` dir to the `/datasets/openimages` then you are ready to go.

## Simple Download

You can download three dataset using the following scripts:

```
curl -L -C - "https://cn-beijing-data.aliyundrive.net/ewoNZgVa%2F629169990%2F6428119fcc0c74f762394d688df3729c210cf3ad%2F6428119f9dd4357d48d94973b04d9d6a1f04ebcb?di=bj29&dr=629169990&f=6428119fcc0c74f762394d688df3729c210cf3ad&pds-params=%7B%22ap%22%3A%2225dzX3vbYqktVxyX%22%7D&security-token=CAIS%2BgF1q6Ft5B2yfSjIr5fXKfjC2ZFk0%2FvSNkyEokIWa8RU3rX%2FqTz2IHFPeHJrBeAYt%2FoxmW1X5vwSlq5rR4QAXlDfNRy7YU76qFHPWZHInuDox55m4cTXNAr%2BIhr%2F29CoEIedZdjBe%2FCrRknZnytou9XTfimjWFrXWv%2Fgy%2BQQDLItUxK%2FcCBNCfpPOwJms7V6D3bKMuu3OROY6Qi5TmgQ41Uh1jgjtPzkkpfFtkGF1GeXkLFF%2B97DRbG%2FdNRpMZtFVNO44fd7bKKp0lQLukMWr%2Fwq3PIdp2ma447NWQlLnzyCMvvJ9OVDFyN0aKEnH7J%2Bq%2FzxhTPrMnpkSlacGoABmWHtcnmwzKp9L7Za%2FRqFuKC32OMUP8p8tNv%2F7SG6yeG6NDhqUAOSs%2FIYBS8flLYQdsjc7hID1hfhF%2FkopFJgGKhQ%2Bw1ksfnRz0Z10%2FTWVBWYurg2xplsdN7RAGv87ayLQKLBH5MPCPKwCDBiCWV8xM2c9AyAaBk5NJ5dVbgZmRogAA%3D%3D&u=899b7467ef024ccb927572b8b4db394f&x-oss-access-key-id=STS.NTbbBv4NEd994j5FBCgKx1pTK&x-oss-additional-headers=referer&x-oss-expires=1702149715&x-oss-signature=K%2FWlyxvUQxOpehy2S6eFG5ifvnhAn%2BHKNEreFWpyQT4%3D&x-oss-signature-version=OSS2" -o "datasets.zip" -e "https://www.aliyundrive.com/"
```

