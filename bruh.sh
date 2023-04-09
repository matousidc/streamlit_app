echo "bruh this shit"
while true; do
    read -p "do you wish to continue?" yn
    case $yn in
        [y]* ) break;;
        [n]* ) exit;;
    esac
done
echo 'continuing'
