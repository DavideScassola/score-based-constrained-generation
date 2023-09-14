
for F in LI NW SNR
do
echo $F
python main-generate.py config/constrained_generation/white_wine_$F.py
done