#include <stdio.h>



/*��������
ʱ�临�Ӷȣ�
    ��ã�O(nlog_2^n)
    ���O(n^2)
    ƽ����O(nlog_2^n)

�ռ临�Ӷȣ� O(nlog_2^n)

���ȶ�

*/




void show_array(int *data, int len){
    for(int i = 0; i < len; ++i){
        printf("%d ", data[i]);
    }
}


int find_pose(int *data, int low, int high){
    int val = data[low];
    while(low < high){
        while(low < high && data[high] >= val){
            --high;
        }
        data[low] = data[high];

        while(low < high && data[low] <= val){
            ++low;
        }
        data[high] = data[low];
    }
    data[low] = val;

   return high; // return low
}


void quick_sort(int * data, int low, int high){
    int pos;
    if (low < high){
        pos = find_pose(data, low, high);
        quick_sort(data, low, pos-1);
        quick_sort(data, pos+1, high);
    }

}



int main(){

    int data[] = {13, 18, 21, 9, 0, 17, 23, 91, 73, 8};
    printf("size:%d\n", sizeof(data) / sizeof(data[0]));
    int len = sizeof(data) / sizeof(data[0]);

    quick_sort(data, 0, len-1);

    show_array(data, len);

    system("pause");

    return 0;
}