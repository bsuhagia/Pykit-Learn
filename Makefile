test:
	@nosetests -a !slow

test-all:
	@nosetests

clean:
	@find . -name *.pyc -type f -delete
	@find . -name cl_gui.log -type f -delete
