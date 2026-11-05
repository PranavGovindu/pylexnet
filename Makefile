CC ?= cc
CFLAGS ?= -O2

pylexnet: pylexnet.c pylexnet_kernel.h
	$(CC) $(CFLAGS) -o $@ pylexnet.c -lm

# Regenerate the weights and kernel from the design. Needs Python and Pygments 2.17.2.
regen:
	python3 compile.py --matcher add

# Check the fast path against the generic evaluation of the same weights.
verify: pylexnet
	python3 tools/verify.py dev

clean:
	rm -f pylexnet

.PHONY: regen verify clean
