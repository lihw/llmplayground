/**
* Misc definitions and helper structures
* Author: lihw81@gmail.com
*/


#ifndef M_DEFS_H
#define M_DEFS_H

#define M_NO_COPY_CONSTRUCTOR(Clazz) \
    Clazz(const Clazz&) = delete; \
    Clazz(Clazz&&) = delete; 
#define M_NO_MOVE_CONSTRUCTOR(Clazz) \
    const Clazz& operator=(const Clazz&) = delete; \
    const Clazz& operator=(Clazz&) = delete; 

#define M_BEGIN_NAMESPACE namespace m {

#define M_END_NAMESPACE };

#endif //! M_DEFS_H
