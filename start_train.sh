
for i in 1 2 3 4 5
do
  echo "Looping ... number $i"
  python train.py
  python train_eval.py
done