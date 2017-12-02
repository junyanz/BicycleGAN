FILE=$1

echo "Note: available models are edges2shoes and facades_label2photo"
echo "downloading [$FILE]"

MODEL_FILE_G=./pretrained_models/${FILE}_net_G.pth
URL_G=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/${FILE}_net_G.pth
wget -N $URL_G -O $MODEL_FILE_G


MODEL_FILE_E=./pretrained_models/${FILE}_net_E.pth
URL_E=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/${FILE}_net_E.pth
wget -N $URL_E -O $MODEL_FILE_E
