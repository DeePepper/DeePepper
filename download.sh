FILE=$1
if [ $FILE == "data" ]; then

    URL=https://www.dropbox.com/s/l49kz27vjdqm781/data.zip?dl=0
    ZIP_FILE=./data/data.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE
fi
