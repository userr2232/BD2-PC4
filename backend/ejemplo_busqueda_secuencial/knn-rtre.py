def searchKNN(node, Q, k)
	if node is leaf:	
		for O in node:
			if listResult.size() < k:
				listResult.push(O, ED(O, Q) )
			else:	
				r = listResult.front().dist				
				if ED(O, Q) <= r:		 # refine		
					listResult.push(O, ED(O, Q) )
					listResult.pop()
	else:
		for child in node:
			if MINDIST(child, Q) <= r:  #filter
				search(child, Q, r)

##########·				
listResult = MaxHeap()
searchKNN(root, Q, k)
