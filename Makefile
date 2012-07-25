default: linux

linux:
	$(MAKE) -C src
	$(MAKE) -C example

macos:
	$(MAKE) -C src
	$(MAKE) -C example

win32:
	$(MAKE) -C src -f Makefile.mingw
	ARCH=32 $(MAKE) -C example -f Makefile.mingw

win64:
	$(MAKE) -C src -f Makefile.mingw
	ARCH=64 $(MAKE) -C example -f Makefile.mingw

clean:
	$(MAKE) -C src clean
	$(MAKE) -C example clean
