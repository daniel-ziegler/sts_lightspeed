//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_ACTIONQUEUE_H
#define STS_LIGHTSPEED_ACTIONQUEUE_H

#include "sts_common.h"

#include <bitset>
#include <functional>
#include <cassert>

#include "combat/Actions.h"


namespace sts {

    class BattleContext;

    // Simple deque
    template<int capacity>
    struct ActionQueue {
        friend BattleContext;
        int front = 0;
        int back = 0;
        int size = 0;

#ifdef sts_action_queue_use_raw_array
        Action arr[capacity];
        ActionQueue() = default;
        ActionQueue(const ActionQueue &rhs) : size(rhs.size), back(rhs.back), front(rhs.front), bits(rhs.bits) {
            int cur = rhs.front;
            for (int i = 0; i < rhs.size; ++i) {
                if (cur >= capacity) {
                    cur = 0;
                }
                arr[cur] = rhs.arr[cur];
            }
        }
#else
        std::array<Action, capacity> arr;
#endif

        void clear();
        void pushFront(Action a);
        void pushBack(Action a);
        bool isEmpty();
        Action popFront();
        [[nodiscard]] int getCapacity() const;
        Action& operator[](int index);
        const Action& operator[](int index) const;
    };

    template<int capacity>
    void ActionQueue<capacity>::clear() {
        size = 0;
        back = 0;
        front = 0;
    }

    template <int capacity>
    void ActionQueue<capacity>::pushFront(Action a) {
#ifdef sts_asserts
        assert(size != capacity);
#endif
        --front;
        ++size;
        if (front < 0) {
            front = capacity-1;
        }
        arr[front] = std::move(a);
    }

    template<int capacity>
    void ActionQueue<capacity>::pushBack(Action a) {
#ifdef sts_asserts
        if (size >= capacity) {
            assert(false);
        }
#endif
        if (back >= capacity) {
            back = 0;
        }
        arr[back] = std::move(a);
        ++back;
        ++size;
    }

    template<int capacity>
    bool ActionQueue<capacity>::isEmpty() {
        return size == 0;
    }

    template<int capacity>
    Action ActionQueue<capacity>::popFront() {
#ifdef sts_asserts
        assert(size > 0);
#endif
        Action a = arr[front];
        ++front;
        --size;
        if (front >= capacity) {
            front = 0;
        }
        return a;
    }

    template<int capacity>
    int ActionQueue<capacity>::getCapacity() const {
        return capacity;
    }

    template<int capacity>
    Action& ActionQueue<capacity>::operator[](int index) {
#ifdef sts_asserts
        assert(0 <= index && index < size);
#endif
        return arr[(front + index) % capacity];
    }

    template<int capacity>
    const Action& ActionQueue<capacity>::operator[](int index) const {
#ifdef sts_asserts
        assert(0 <= index && index < size);
#endif
        return arr[(front + index) % capacity];
    }
    
    template<int capacity> std::ostream& operator<<(std::ostream &os, const ActionQueue<capacity> &queue) {
        os << "[";
        for (int i = 0; i < queue.size; i++) {
            if (i > 0) {
                os << ", ";
            }
            os << queue[i];
        }
        os << "]";
        return os;
    }

}


#endif //STS_LIGHTSPEED_ACTIONQUEUE_H
