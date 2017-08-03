1	def id3( data, attributes, default)  
2		if data is empty, return default  
3		if data is homogeneous, return class label.  
4		if attributes is empty, return majority-label( data)  
5		best_attr = pick_best_attribute( data, attributes)  
6		node = new Node( best_attribute)  
7		default_label = majority-label( data)  
8		for value in the domain of best_attr  
9 			subset = examples in data where best_attr == value  
10			child = id3( subset, attributes - best_attr, default_label)  
11			add child to node  
12		end  
13		return node  
14	end  
  
Comments
As far as implementation details go, you need to provide functions that will test:  
1. Does the data have all the same label?  
2. What is the majority label?  
3. What is the best attribute?  
4. Given an attribute and a value from its domain, return the examples in the data where that attribute has that value.  

You could create a Node class...but since there are no methods, it's more of a struct and you can just use a Dict (HashTable) if you like.  
You must take care in the recursive part. "attributes - best_attr" needs to be a new list (deepcopy) of attributes with best_attr removed.  

---

function DECISION-TREE-LEARNING(_examples_, _attributes_, _parent_examples_) returns a tree  
	*if* _examples_ is empty *then* *return* PLURALITY-VALUE(_parent_examples_)  
	*else* *if* all examples have the same classification *then* *return* the classification  
	*else* *if* attributes is empty *then* *return* P LURALITY -V ALUE (examples)  
	*else*  
		A <- argmax(a of all attributes)[IMPORTANCE(a,examples)]  
		_tree_ <- a new decision tree with root test A  
		*for* *each* value v of A *do*  
			_exs_ <- {e: e in examples *and* e.A = v }  
			_subtree_ <- DECISION-TREE-LEARNING(exs, attributes-A, examples)  
			add a branch to _tree_ with label (A = v) and subtree _subtree_  
		*return* _tree_  
