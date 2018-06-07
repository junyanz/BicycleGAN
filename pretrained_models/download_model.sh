FILE=$1

echo "Note: available models are edges2shoes, edges2handbags, night2day, maps, and facades"
echo "downloading [$FILE]"
MODEL_DIR=./pretrained_models/${FILE}
mkdir -p ${MODEL_DIR}


MODEL_FILE_G=${MODEL_DIR}/latest_net_G.pth
URL_G=http://efrosgans.eecs.berkeley.edu/BicycleGAN//models/${FILE}_net_G.pth
wget -N $URL_G -O $MODEL_FILE_G


MODEL_FILE_E=${MODEL_DIR}/latest_net_E.pth
URL_E=http://efrosgans.eecs.berkeley.edu/BicycleGAN//models/${FILE}_net_E.pth
wget -N $URL_E -O $MODEL_FILE_E
