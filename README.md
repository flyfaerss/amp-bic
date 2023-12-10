## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

You can download three dataset from [[download_bash](curl -L -C - "https://cn-beijing-data.aliyundrive.net/ewoNZgVa%2F629169990%2F6428119fcc0c74f762394d688df3729c210cf3ad%2F6428119f9dd4357d48d94973b04d9d6a1f04ebcb?di=bj29&dr=629169990&f=6428119fcc0c74f762394d688df3729c210cf3ad&pds-params=%7B%22ap%22%3A%2225dzX3vbYqktVxyX%22%7D&security-token=CAIS%2BgF1q6Ft5B2yfSjIr5fXKfjC2ZFk0%2FvSNkyEokIWa8RU3rX%2FqTz2IHFPeHJrBeAYt%2FoxmW1X5vwSlq5rR4QAXlDfNRy7YU76qFHPWZHInuDox55m4cTXNAr%2BIhr%2F29CoEIedZdjBe%2FCrRknZnytou9XTfimjWFrXWv%2Fgy%2BQQDLItUxK%2FcCBNCfpPOwJms7V6D3bKMuu3OROY6Qi5TmgQ41Uh1jgjtPzkkpfFtkGF1GeXkLFF%2B97DRbG%2FdNRpMZtFVNO44fd7bKKp0lQLukMWr%2Fwq3PIdp2ma447NWQlLnzyCMvvJ9OVDFyN0aKEnH7J%2Bq%2FzxhTPrMnpkSlacGoABmWHtcnmwzKp9L7Za%2FRqFuKC32OMUP8p8tNv%2F7SG6yeG6NDhqUAOSs%2FIYBS8flLYQdsjc7hID1hfhF%2FkopFJgGKhQ%2Bw1ksfnRz0Z10%2FTWVBWYurg2xplsdN7RAGv87ayLQKLBH5MPCPKwCDBiCWV8xM2c9AyAaBk5NJ5dVbgZmRogAA%3D%3D&u=899b7467ef024ccb927572b8b4db394f&x-oss-access-key-id=STS.NTbbBv4NEd994j5FBCgKx1pTK&x-oss-additional-headers=referer&x-oss-expires=1702149715&x-oss-signature=K%2FWlyxvUQxOpehy2S6eFG5ifvnhAn%2BHKNEreFWpyQT4%3D&x-oss-signature-version=OSS2" -o "datasets.zip" -e "https://www.aliyundrive.com/")]

## Training 

### Scene Graph Generation Model
In scripts/, we provide train and test instruction templates for VG, OI V6, GQA-LT.

We also provide the trained models (.pth and .sh) in [[model_zoo](https://pan.baidu.com/s/1kndK-j6FhZl7kmh14h5Uxg?pwd=69db)], which include the main results reported in our paper.



## Acknowledgment

This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)