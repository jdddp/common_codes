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
        case 'listCategories':
            return await listCategories(event);
        case 'createCategory':
            return await createCategory(event, wxContext);
        case 'updateCategory':
            return await updateCategory(event, wxContext);
        case 'listDevices':
            return await listDevices(event);
        case 'createDevice':
            return await createDevice(event, wxContext);
        case 'updateDevice':
            return await updateDevice(event, wxContext);
        case 'updateDeviceQuantity':
            return await updateDeviceQuantity(event, wxContext);
        case 'disableDevice':
            return await disableDevice(event, wxContext);
        case 'enableDevice':
            return await enableDevice(event, wxContext);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function checkAdmin(wxContext) {
    const token = wxContext.token || wxContext.TOKEN;
    if (!token) {
        return { code: -1, message: '未登录' };
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, 'your-jwt-secret-key');

    const userResult = await db.collection('user').doc(decoded.userId).get();
    if (!userResult.data || userResult.data.role !== 'admin') {
        return { code: -1, message: '无管理员权限' };
    }

    return { code: 0, userId: decoded.userId };
}

async function listCategories(event) {
    const { parentId = null, level = 1 } = event;

    try {
        let query = db.collection('category');

        if (parentId) {
            query = query.where({ parentId });
        } else {
            query = query.where({ parentId: _.in([null, '']) });
        }

        const result = await query
            .orderBy('sort', 'asc')
            .orderBy('createdAt', 'asc')
            .get();

        return { code: 0, data: result.data };
    } catch (error) {
        console.error('获取分类列表失败:', error);
        return { code: -1, message: '获取分类列表失败' };
    }
}

async function createCategory(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { name, parentId, level, sort } = event;

    if (!name) {
        return { code: -1, message: '分类名称不能为空' };
    }

    try {
        const existingCategory = await db.collection('category')
            .where({
                name,
                parentId: parentId || ''
            })
            .limit(1)
            .get();

        if (existingCategory.data.length > 0) {
            return { code: -1, message: '分类名称已存在' };
        }

        const result = await db.collection('category').add({
            data: {
                name,
                parentId: parentId || '',
                level: level || 1,
                sort: sort || 0,
                status: 'active',
                createdAt: db.serverDate(),
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '创建成功', data: { _id: result._id } };
    } catch (error) {
        console.error('创建分类失败:', error);
        return { code: -1, message: '创建分类失败' };
    }
}

async function updateCategory(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { categoryId, name, sort } = event;

    if (!categoryId) {
        return { code: -1, message: '分类ID不能为空' };
    }

    try {
        const updateData = {
            updatedAt: db.serverDate()
        };

        if (name) updateData.name = name;
        if (sort !== undefined) updateData.sort = sort;

        await db.collection('category').doc(categoryId).update({
            data: updateData
        });

        return { code: 0, message: '更新成功' };
    } catch (error) {
        console.error('更新分类失败:', error);
        return { code: -1, message: '更新分类失败' };
    }
}

async function listDevices(event) {
    const { categoryId, page = 1, pageSize = 20, status = 'active' } = event;

    try {
        let query = db.collection('device_type');

        if (categoryId) {
            query = query.where({ categoryId });
        }

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

        const devices = result.data;

        for (let device of devices) {
            const borrowResult = await db.collection('borrow_record')
                .where({
                    deviceTypeId: device._id,
                    status: _.in(['borrowing', 'partially_returned'])
                })
                .get();

            let borrowedQuantity = 0;
            for (let record of borrowResult.data) {
                borrowedQuantity += record.quantity - record.returnedQuantity;
            }

            device.borrowedQuantity = borrowedQuantity;
            device.availableQuantity = device.totalQuantity - borrowedQuantity;
        }

        return {
            code: 0,
            data: {
                list: devices,
                total,
                page,
                pageSize
            }
        };
    } catch (error) {
        console.error('获取设备列表失败:', error);
        return { code: -1, message: '获取设备列表失败' };
    }
}

async function createDevice(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { name, categoryId, totalQuantity, remark } = event;

    if (!name || !categoryId) {
        return { code: -1, message: '设备名称和分类不能为空' };
    }

    if (totalQuantity < 0) {
        return { code: -1, message: '总数量不能小于0' };
    }

    try {
        const existingDevice = await db.collection('device_type')
            .where({
                name,
                categoryId
            })
            .limit(1)
            .get();

        if (existingDevice.data.length > 0) {
            return { code: -1, message: '设备名称已存在' };
        }

        const result = await db.collection('device_type').add({
            data: {
                name,
                categoryId,
                totalQuantity: totalQuantity || 0,
                status: 'active',
                remark: remark || '',
                createdAt: db.serverDate(),
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '创建成功', data: { _id: result._id } };
    } catch (error) {
        console.error('创建设备失败:', error);
        return { code: -1, message: '创建设备失败' };
    }
}

async function updateDevice(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { deviceId, name, categoryId, remark } = event;

    if (!deviceId) {
        return { code: -1, message: '设备ID不能为空' };
    }

    try {
        const updateData = {
            updatedAt: db.serverDate()
        };

        if (name) updateData.name = name;
        if (categoryId) updateData.categoryId = categoryId;
        if (remark !== undefined) updateData.remark = remark;

        await db.collection('device_type').doc(deviceId).update({
            data: updateData
        });

        return { code: 0, message: '更新成功' };
    } catch (error) {
        console.error('更新设备失败:', error);
        return { code: -1, message: '更新设备失败' };
    }
}

async function updateDeviceQuantity(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { deviceId, totalQuantity } = event;

    if (!deviceId) {
        return { code: -1, message: '设备ID不能为空' };
    }

    if (totalQuantity === undefined || totalQuantity < 0) {
        return { code: -1, message: '总数量不能小于0' };
    }

    try {
        const deviceResult = await db.collection('device_type').doc(deviceId).get();
        if (!deviceResult.data) {
            return { code: -1, message: '设备不存在' };
        }

        const device = deviceResult.data;

        const borrowResult = await db.collection('borrow_record')
            .where({
                deviceTypeId: deviceId,
                status: _.in(['borrowing', 'partially_returned'])
            })
            .get();

        let borrowedQuantity = 0;
        for (let record of borrowResult.data) {
            borrowedQuantity += record.quantity - record.returnedQuantity;
        }

        if (totalQuantity < borrowedQuantity) {
            return {
                code: -1,
                message: `总数量不能小于已借出数量（${borrowedQuantity}）`
            };
        }

        await db.collection('device_type').doc(deviceId).update({
            data: {
                totalQuantity,
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '更新成功' };
    } catch (error) {
        console.error('更新设备数量失败:', error);
        return { code: -1, message: '更新设备数量失败' };
    }
}

async function disableDevice(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { deviceId } = event;

    if (!deviceId) {
        return { code: -1, message: '设备ID不能为空' };
    }

    try {
        await db.collection('device_type').doc(deviceId).update({
            data: {
                status: 'disabled',
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '禁用成功' };
    } catch (error) {
        console.error('禁用设备失败:', error);
        return { code: -1, message: '禁用设备失败' };
    }
}

async function enableDevice(event, wxContext) {
    const adminCheck = await checkAdmin(wxContext);
    if (adminCheck.code !== 0) return adminCheck;

    const { deviceId } = event;

    if (!deviceId) {
        return { code: -1, message: '设备ID不能为空' };
    }

    try {
        await db.collection('device_type').doc(deviceId).update({
            data: {
                status: 'active',
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '启用成功' };
    } catch (error) {
        console.error('启用设备失败:', error);
        return { code: -1, message: '启用设备失败' };
    }
}
