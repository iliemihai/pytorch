.PHONY: clean data requirements 

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	pipenv install -r requirements.txt

data: requirements

clean:
	find . -name "*.pyc" -exec rm {} \;

