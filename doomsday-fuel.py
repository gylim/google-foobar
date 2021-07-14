from fractions import Fraction

'''Unfortunately this solution doesn't pass Google's internal tests, but works in Python 3. 
This solution implements the approach described here: http://www-personal.umd.umich.edu/~fmassey/math420/Notes/c2/2.6.1%20Probability%20of%20Reaching%20a%20State.doc
The gcd, lcm and gauss_elim functions are taken from https://surajshetiya.github.io/Google-foobar/, while
the copy_matrix and matrix_subtraction code is adapted from https://integratedmlai.com/basic-linear-algebra-tools-in-pure-python-without-numpy-or-scipy/
'''

# Find greatest common divisor for 2 numbers
def gcd(x, y):
    def gcd1(x, y):
        if (y == 0):
            return x
        return gcd1(y, x%y)
    return gcd1(abs(x), abs(y))

# Find lowest common multiple for 2 numbers
def lcm(x, y):
    return int(x*y/gcd(x,y))

# copy the transition matrix in the gauss_elim function to prevent the values from changing in the 
# second for loop of the solution
def copy_matrix(M):
    n = len(M)
    MC = [[0]*n for num in range(n)]
    for i in range(n):
        for j in range(n):
            MC[i][j] = M[i][j]
    return MC

# creates an identity matrix with ones on the diagonal and zeros elsewhere
def identity(n):
    i_matrix = [[0]*n for num in range(n)]
    for i in range(n):
        i_matrix[i][i] = 1
    return i_matrix

# element-wise subtraction of 2 matrices without numpy
def matrix_subtraction(A, B):
    C = [[0]*len(A) for num in range(len(A))]
 
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] - B[i][j]
    return C

# gaussian elimination for solving linear equations with fractions
def gauss_elim(matrix, values):
    mat = copy_matrix(matrix)
    for i in range(len(mat)):
        index = -1
        for j in range(i, len(mat)):
            if (mat[j][i].numerator != 0):
                index = j
                break
        if (index == -1):
            raise ValueError('Gauss elimination failed!')
        mat[i], mat[index] = mat[index], mat[j]
        values[i], values[index] = values[index], values[i]
        for j in range(i+1, len(mat)):
            if (mat[j][i].numerator == 0):
                continue
            ratio = -mat[j][i]/mat[i][i]
            for k in range(i, len(mat)):
                mat[j][k] += ratio * mat[i][k]
            values[j] += ratio * values[i]
    res = [0 for i in range(len(mat))]
    for i in range(len(mat)):
        index = len(mat) -1 -i
        end = len(mat) - 1
        while end > index:
            values[index] -= mat[index][end] * res[end]
            end -= 1
        res[index] = values[index]/mat[index][index]
    return res[0]

def solution(a):
    final_states = 0
    first = []
    # split array list into absorbing and initial/transition states
    for i in a:
        if (sum(i) > 0):
            first.append([Fraction(num/sum(i)).limit_denominator(100) for num in i])
        else:
            final_states += 1
    x = len(a) - final_states
    # create I and PT matrices from existing lists
    id_mat, second = identity(x), [num[:x] for num in first]
    # derive I-PT matrix
    trans_mat = matrix_subtraction(id_mat, second)
    count = final_states
    raw_ans = []
    # solve linear equations to get F0 for each absorbing state
    for l in range(final_states):
        figures = [row[-count] for row in first]
        raw_ans.append(gauss_elim(trans_mat, figures))
        count -= 1
    # change fractions to same denominator, extract denominator
    l = 1
    for item in raw_ans:
        l = lcm(l, item.denominator)
    answer = list(map(lambda x: int(x.numerator*l/x.denominator), raw_ans))
    answer.append(l)
    return answer