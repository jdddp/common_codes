const cloud = require('wx-server-sdk');

cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const _ = db.command;

exports.main = async (event, context) => {
    const { action } = event;
    const wxContext = cloud.getWXContext();

    switch (action) {
        case 'returnDevice':
            return await returnDevice(event, wxContext);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function getCurrentUser(event) {
    const token = event.token;
    if (!token) {
        return { code: -1, message: '未登录' };
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, 'A2CF692946145E42362A1BA63DAAC972457CC0CC6ED9B7D7FCD2FB250F33B7F7');

    const userResult = await db.collection('user').doc(decoded.userId).get();
    if (!userResult.data) {
        return { code: -1, message: '用户不存在' };
    }

    const user = userResult.data;

    if (user.status === 'disabled') {
        return { code: -1, message: '账号已被禁用' };
    }

    if (user.passwordVersion !== decoded.passwordVersion) {
        return { code: -1, message: '登录已失效，请重新登录' };
    }

    return { code: 0, user };
}

async function returnDevice(event, wxContext) {
    const { borrowRecordId, quantity, condition, remark } = event;

    if (!borrowRecordId) {
        return { code: -1, message: '借用记录ID不能为空' };
    }

    if (!quantity || quantity <= 0) {
        return { code: -1, message: '归还数量必须大于0' };
    }

    const userCheck = await getCurrentUser(event);
    if (userCheck.code !== 0) return userCheck;

    const user = userCheck.user;

    try {
        const borrowResult = await db.collection('borrow_record').doc(borrowRecordId).get();
        if (!borrowResult.data) {
            return { code: -1, message: '借用记录不存在' };
        }

        const borrowRecord = borrowResult.data;

        if (borrowRecord.userId !== user._id) {
            return { code: -1, message: '只能归还自己的借用记录' };
        }

        if (borrowRecord.status === 'returned' || borrowRecord.status === 'cancelled') {
            return { code: -1, message: '该记录已归还或已取消' };
        }

        const outstandingQuantity = borrowRecord.quantity - borrowRecord.returnedQuantity;

        if (quantity > outstandingQuantity) {
            return {
                code: -1,
                message: `归还数量不能超过待归还数量（${outstandingQuantity}）`
            };
        }

        const returnResult = await db.collection('return_record').add({
            data: {
                borrowRecordId,
                userId: user._id,
                deviceTypeId: borrowRecord.deviceTypeId,
                quantity,
                returnTime: db.serverDate(),
                condition: condition || 'normal',
                remark: remark || '',
                createdAt: db.serverDate()
            }
        });

        const newReturnedQuantity = borrowRecord.returnedQuantity + quantity;
        let newStatus = 'partially_returned';
        if (newReturnedQuantity >= borrowRecord.quantity) {
            newStatus = 'returned';
        }

        await db.collection('borrow_record').doc(borrowRecordId).update({
            data: {
                returnedQuantity: newReturnedQuantity,
                status: newStatus,
                updatedAt: db.serverDate()
            }
        });

        return {
            code: 0,
            message: '归还成功',
            data: {
                _id: returnResult._id,
                newStatus
            }
        };
    } catch (error) {
        console.error('归还设备失败:', error);
        return { code: -1, message: '归还设备失败' };
    }
}
