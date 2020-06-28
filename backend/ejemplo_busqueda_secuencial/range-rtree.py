def search(node, Q, r)
	if node is leaf:	
		for O in node: 
			if ED(O, Q) <= r: # refine
				add(listResult, o)
	else:
		for child in node:
			if MINDIST(child, Q) <= r:  #filter
				search(child, Q, r)
	
##########·				
listResult = []
search(root, Q, r)	


				
			
