#import deque

def binary_search(arr, target):
    first, last = 0, len(arr) - 1

    while first < last:
        mid = (first + last) // 2
        if arr[mid] == target:
            return True
        
        else:
            if arr[mid] < target:
                last = mid +1
            elif arr[mid] > target:
                first = mid - 1

    return False

def main():
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    print(max_area(heights)) 
    return None

if __name__ == "__main__":
    print(main())