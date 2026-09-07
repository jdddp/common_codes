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
        case 'borrowDevice':
            return await borrowDevice(event, wxContext);
        case 'listMyBorrowings':
            return await listMyBorrowings(event, wxContext);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function getCurrentUser(wxContext) {
    const token = wxContext.token || wxContext.TOKEN;
    if (!token) {
        return { code: -1, message: '未登录' };
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, 'your-jwt-secret-key');

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

async function borrowDevice(event, wxContext) {
    const { deviceTypeId, quantity, expectedReturnTime, purpose, remark } = event;

    if (!deviceTypeId) {
        return { code: -1, message: '设备类型不能为空' };
    }

    if (!quantity || quantity <= 0) {
        return { code: -1, message: '借用数量必须大于0' };
    }

    const userCheck = await getCurrentUser(wxContext);
    if (userCheck.code !== 0) return userCheck;

    const user = userCheck.user;

    try {
        const deviceResult = await db.collection('device_type').doc(deviceTypeId).get();
        if (!deviceResult.data) {
            return { code: -1, message: '设备不存在' };
        }

        const device = deviceResult.data;

        if (device.status === 'disabled') {
            return { code: -1, message: '该设备当前不可借用' };
        }

        const borrowResult = await db.collection('borrow_record')
            .where({
                deviceTypeId,
                status: _.in(['borrowing', 'partially_returned'])
            })
            .get();

        let borrowedQuantity = 0;
        for (let record of borrowResult.data) {
            borrowedQuantity += record.quantity - record.returnedQuantity;
        }

        const availableQuantity = device.totalQuantity - borrowedQuantity;

        if (quantity > availableQuantity) {
            return {
                code: -1,
                message: `当前仅剩 ${availableQuantity} 台可用设备`
            };
        }

        const result = await db.collection('borrow_record').add({
            data: {
                userId: user._id,
                deviceTypeId,
                quantity,
                returnedQuantity: 0,
                status: 'borrowing',
                borrowTime: db.serverDate(),
                expectedReturnTime: expectedReturnTime || null,
                purpose: purpose || '',
                remark: remark || '',
                createdAt: db.serverDate(),
                updatedAt: db.serverDate()
            }
        });

        return {
            code: 0,
            message: '借用成功',
            data: { _id: result._id }
        };
    } catch (error) {
        console.error('借用设备失败:', error);
        return { code: -1, message: '借用设备失败' };
    }
}

async function listMyBorrowings(event, wxContext) {
    const { page = 1, pageSize = 20, status = '' } = event;

    const userCheck = await getCurrentUser(wxContext);
    if (userCheck.code !== 0) return userCheck;

    const user = userCheck.user;

    try {
        let query = db.collection('borrow_record')
            .where({ userId: user._id });

        if (status) {
            query = query.where({ status });
        }

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        const borrowings = result.data;

        for (let borrowing of borrowings) {
            const deviceResult = await db.collection('device_type')
                .doc(borrowing.deviceTypeId)
                .get();

            if (deviceResult.data) {
                borrowing.deviceName = deviceResult.data.name;
            }
        }

        return {
            code: 0,
            data: {
                list: borrowings,
                total,
                page,
                pageSize
            }
        };
    } catch (error) {
        console.error('获取借用列表失败:', error);
        return { code: -1, message: '获取借用列表失败' };
    }
}
