FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/testset/${FILE}.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
