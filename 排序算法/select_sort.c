#include <stdio.h>
#include <malloc.h>


/*  选择排序
时间: O(n^2)
空间：O(1)
不稳定



*/


void show_array(int *data, int len){
    for(int i = 0; i < len; ++i){
        printf("%d ", data[i]);
    }
}


void select_sort(int *data, int len){
    int  min_ind, tmp;
    for(int i = 0; i < len; ++i){
        min_ind = i;
        for(int j = i+1; j < len; j++){
            if(data[j] < data[min_ind]){
                min_ind = j;
            }
        }

        if(min_ind != i){
            tmp = data[min_ind];
            data[min_ind] = data[i];
            data[i] = tmp;
        }
    }
}


int main(){

    printf("????!\n");

    int data[] = {13, 18, 21, 9, 0, 17, 23, 91, 73, 8};
    printf("size:%d\n", sizeof(data) / sizeof(data[0]));
    int len = sizeof(data) / sizeof(data[0]);

    select_sort(data, len);
    show_array(data, len);

    system("pause");

    return 0;
}



















