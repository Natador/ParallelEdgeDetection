CC = cc
task4: task4.c
	$(CC) -o $@ $^

clean:
	rm task4
