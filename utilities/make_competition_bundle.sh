DATE=`date "+%Y-%m-%d-%H-%M-%S"`
DIR='tmp/'
# Delete $DIR if exists
rm -rf $DIR
mkdir $DIR
cp '../'*.jpg $DIR
cp '../'*.html $DIR
cp '../'*.yaml $DIR
cp -r '../assets' $DIR
cd .. # bundle/

# Begin zipping each folder
for filename in reference_data_1 reference_data_2 scoring_program input
do
  cd $filename
  echo 'Zipping: '$filename
  zip -o -r --exclude=*.git* --exclude=*__pycache__* --exclude=*.DS_Store* --exclude=*public_data* "../utilities/"$DIR$filename .;
  cd .. # bundle/
done

# Zip all to make a competition bundle
cd "utilities/"$DIR
zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../bundle_"$DATE .;
