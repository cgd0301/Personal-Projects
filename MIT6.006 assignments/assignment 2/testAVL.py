class AVLTree:
    def __init__(self, root):
        self.root = root #Node
        self.len = 0
    
    def append(self, value):
        self.len += 1
        ptr = self.root
        if (ptr is None):
            self.root = Node(None, Node(ptr, None, None, None, -1), Node(ptr, None, None, None, -1), value, 0)
            return

        while not ptr.is_leaf():
            if (value >= ptr.value and ptr.right.value != None):
                ptr = ptr.right
            elif (value < ptr.value and ptr.left.value != None):
                ptr = ptr.left
            else:
                break 

        if (value >= ptr.value):
            ptr.right = Node(ptr, Node(ptr.right, None, None, None, -1), Node(ptr.right, None, None, None, -1), value, 0)
        else:
            ptr.left = Node(ptr, Node(ptr.left, None, None, None, -1), Node(ptr.left, None, None, None, -1), value, 0)
        
        cpy_ptr = ptr
        while (ptr is not None):
            ptr.height += 1
            ptr = ptr.parent
        self.balance(cpy_ptr.find_unbalance())
    
    def left_rotate(self, node):
        node_r = node.right
        if (node.is_root()):
            self.root = node_r
        else:
            if (node.parent.left == node):
                node.parent.left = node_r
            else:
                node.parent.right = node_r

        tmp_ptr1 = node_r.left

        node_r.parent = node.parent
        node_r.left = node
        node.right = tmp_ptr1
        node.parent = node_r
        tmp_ptr1.parent = node

    def right_rotate(self, node):
        node_l = node.left
        if (node.is_root()):
            self.root = node_l
        else:
            if (node.parent.right == node):
                node.parent.right = node_l
            else:
                node.parent.left = node_l

        tmp_ptr1 = node_l.right

        node_l.parent = node.parent
        node_l.right = node
        node.left = tmp_ptr1
        node.parent = node_l
        tmp_ptr1.parent = node
        

    #unbalanced node should be the lowest unbalanced node
    def balance(self, unbalanced_node):
        if (unbalanced_node == None or unbalanced_node.is_balanced()):
            return
        
        unbalanced_node_par = unbalanced_node.parent
        r_child = unbalanced_node.right
        l_child = unbalanced_node.left


        if (unbalanced_node.left.height - unbalanced_node.right.height < 0):
            if (r_child.right.height >= r_child.left.height):
                self.left_rotate(unbalanced_node)
                self.update_height(unbalanced_node)
            else:
                self.right_rotate(r_child)
                self.update_height(r_child)
                self.left_rotate(unbalanced_node)
                self.update_height(unbalanced_node)

        else:
            if (l_child.left.height >= l_child.right.height):
                self.right_rotate(unbalanced_node)
                self.update_height(unbalanced_node)
            else:
                self.left_rotate(l_child)
                self.update_height(l_child)
                self.right_rotate(unbalanced_node)
                self.update_height(unbalanced_node)

        self.balance(unbalanced_node_par)

    def min(self):
        ptr = self.root

        if (ptr is None):
            return None

        while (ptr.left.value is not None):
            ptr = ptr.left

        return ptr.value

    def pop(self):
        ptr = self.root

        if (ptr is None):
            return None
        
        self.len -= 1

        while (ptr.left.value is not None):
            ptr = ptr.left

        min_par = ptr.parent
        min_value = ptr.value
        if (min_par is not None):
            min_par.left = ptr.right
        else:
            if (ptr.right.value is not None):
                self.root = ptr.right
            else:
                self.root = None
            return ptr.value
        ptr.right.parent = min_par

        self.update_height(min_par)
        if (min_par is not None):
            self.balance(min_par.find_unbalance())

        return min_value
    
    def update_height(self, node):
        if (node is None):
            return
        
        if (node.left.height <= node.right.height):
            node.height = node.right.height + 1
        else:
            node.height = node.left.height + 1

        self.update_height(node.parent)

   
    def delete(self, node):
        if (node is None):
            return
        
        if (node.is_leaf()):
            if (node.parent is None):
                self.root = None
            else:
                if (node == node.parent.left):
                    node.parent.left = Node(node.parent, None, None, None, -1)
                else:
                    node.parent.right = Node(node.parent, None, None, None, -1)
        elif (node.left.value is not None and node.right.value is None):
            node.left.parent = node.parent
            if (node.parent is not None):
                if (node == node.parent.left):
                    node.parent.left = node.left
                else:
                    node.parent.right = node.left
        elif (node.left.valye is None and node.right.value is not None):
            node.right.parent = node.parent
            if (node.parent is not None):
                if (node == node.parent.left):
                    node.parent.left = node.right
                else:
                    node.parent.right = node.right
        else:
            r_subtree = AVLTree(node.right)
            """
                TODO:
            """

   
class Node:
    def __init__(self, parent, left, right, value, height):
        self.parent = parent
        self.left = left
        self.right = right
        self.value = value
        self.height = height

    def is_leaf(self):
        return (self.left.value is None and self.right.value is None)
    
    def is_balanced(self):
        return (abs(self.left.height - self.right.height) <= 1)
    
    def is_root(self):
        return (self.parent is None)
    
    def print_AVL(self):        
        if (self.is_leaf()):
            print(self.value, self.height, self.left.value, self.right.value)
            return
        
        if (self.left.value is not None):
            self.left.print_AVL()
        
        print(self.value, self.height, self.left.value, self.right.value)
        
        if (self.right.value is not None):
            self.right.print_AVL()

        return

    def find_unbalance(self):
        #find the unbalance node at lowest level
        ptr = self

        while (ptr is not None):
            if (abs(ptr.left.height - ptr.right.height) > 1):
                return ptr
            ptr = ptr.parent
        
        return None
            
    
def main():
    test_AVL = AVLTree(None)
    test_AVL.append(15)
    test_AVL.append(10)
    test_AVL.append(5)
    test_AVL.append(20)
    test_AVL.append(25)
    test_AVL.append(30)
    test_AVL.append(26)
    print("**********")
    test_AVL.root.print_AVL()

    print("**********")
    for i in range(5):
        test_AVL.pop()
        print(test_AVL.len)
        test_AVL.root.print_AVL()
        print("***********")

main()
