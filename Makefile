default: linux

linux:
	$(MAKE) -C src
	$(MAKE) -C example

macos:
	$(MAKE) -C src
	$(MAKE) -C example

win32:
	$(MAKE) -C src -f Makefile.mingw
	$(MAKE) -C example -f Makefile.mingw

clean:
	$(MAKE) -C src clean
	$(MAKE) -C example clean
