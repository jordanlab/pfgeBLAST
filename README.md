# pfgeBLAST
Tool for the rapid inference of O-serogroup and PFGE pattern for Shiga toxin-producing E. coli

## Required software and libraries
* Linux OS
* Python >= 3.7.4
* Scikit-learn >= v.0.22.1
* Pandas >= v1.0.1
* Numpy >= v1.18.1
* Mash >= v2.1

## Quick start
1. Download repository from github
2. Unzip repository and enter directory
> cd pfgeBLAST/
3. Untar directory containing pre-generated models
> tar -xzvf SavedModels.tar.gz
4. Copy desired paired-end FASTQ files into directory "InputFiles". Multiple file pairs can be copied here if running in batch.
5. Run program executable: predict_pfge.py with appropriate flag options
> python predict_pfge.py -m </path/to/mash-executable> -o </path/to/desired/output-directory>