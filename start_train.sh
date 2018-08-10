
for i in 1 2 3 4 5
do
  echo "Looping ... number $i"
  python train_eval.py
  python train.py
  python valid_eval.py
  python lfw_eval.py
done