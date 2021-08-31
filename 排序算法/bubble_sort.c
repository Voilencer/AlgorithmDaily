#include <stdio.h>



/*冒泡排序:从小到大
时间复杂度： O(n^2)
空间复杂度： O(1)
* 特点�?
实现简单，n较小时性能较好

*/


void show_array(int *data, int len){
    for(int i = 0; i < len; ++i){
        printf("%d ", data[i]);
    }
}

void bubble_sort(int *data, int len){
    int tmp;
    for(int i = 0; i < len-1; ++i){
        for(int j = 0; j < len-i-1; ++j){
            if(data[j] > data[j+1]){
                tmp = data[j];
                data[j] = data[j+1];
                data[j+1] = tmp;
            }
        }
    }

    return;
}



int main(){

    printf("bubble sort!\n");

    int data[] = {13, 18, 21, 9, 0, 17, 23, 91, 73, 8};
    printf("size:%d\n", sizeof(data) / sizeof(data[0]));
    int len = sizeof(data) / sizeof(data[0]);

    bubble_sort(data, len);

    show_array(data, len);  

    system("pause");

    return 0;
}