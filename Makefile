CC = cc
CFLAGS = -O3 -lm -g 

learn: learn.out
	echo "created ./learn.out"
learn.out: src/learn.c src/nn.c
	$(CC) $(CFLAGS) -Iinclude -o learn.out $^ -lm

iterate: iterate.out
	echo "created ./iterate.out"
iterate.out: examples/iterate.c src/nn.c
	$(CC) $(CFLAGS) -Iinclude -o iterate.out $^ -lraylib -lm

digits: digits.out
	echo "created ./digits.out"
digits.out: examples/digits.c src/nn.c
	$(CC) $(CFLAGS) -Iinclude -o digits.out $^ -lraylib -lm

grapher: grapher.out
	echo "created ./grapher.out"
grapher.out: examples/grapher.c src/nn.c
	$(CC) $(CFLAGS) -Iinclude -o grapher.out $^ -lraylib -lm
