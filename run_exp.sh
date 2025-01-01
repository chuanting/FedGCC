# baseline
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -no-compressed -ratio 1.0 -file 'trentino' -strategy 'kb' >>fig6.log &
# kb 0.01
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.01 -file 'trentino' -strategy 'kb' >>fig6.log &
# kb 0.05
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.05 -file 'trentino' -strategy 'kb' >>fig6.log &
# kb 0.1
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.1 -file 'trentino' -strategy 'kb' >>fig6.log &

wait
# baseline
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -no-compressed -ratio 1.0 -file 'milano' -strategy 'kb' >>fig6.log &
# kb 0.01
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.01 -file 'milano' -strategy 'kb' >>fig6.log &
# kb 0.05
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.05 -file 'milano' -strategy 'kb' >>fig6.log &
# kb 0.1
CUDA_VISIBLE_DEVICES=0 python main.py -mu 0.1 -comp 'dgc' -compressed -ratio 0.1 -file 'milano' -strategy 'kb' >>fig6.log &