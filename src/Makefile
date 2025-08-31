CC = cc
CFLAGS = -ggdb -g -O3 -lm

main.out: main.c nn.c
	$(CC) $(CFLAGS) -I../include -o main.out $^

digits.out: digits.c nn.c
	$(CC) $(CFLAGS) -I../include -o digits.out $^ -lraylib
