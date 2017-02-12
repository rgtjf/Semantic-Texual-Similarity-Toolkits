
cd ./data
cat STS2016* > input.test.txt
cp *.txt ./eval/
cd ..

sh train.sh -wordstem simlex -wordfile paragram -outfile cpu-word-model -updatewords True -dim 300 -traindata ../data/input.all.txt -devdata ../data/input.test.txt -testdata ../data/SICK_trial.manual.SICK.txt -layersize 300 -save True -nntype proj -numlayers 3 -outgate False -nonlinearity 2 -evaluate True -epochs 5 -minval 0 -maxval 5 -traintype normal -task sim -batchsize 50 -LW 1e-05 -LC 1e-06 -memsize 50 -learner adam -eta 0.001