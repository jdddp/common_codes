const auth = require('../../../utils/auth');
const util = require('../../../utils/util');

Page({
    data: {
        categories: [],
        loading: false
    },

    onLoad() {
        if (!auth.checkAdmin()) {
            return;
        }
        this.loadCategories();
    },

    onShow() {
        this.loadCategories();
    },

    async loadCategories() {
        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('device', {
                action: 'listCategories'
            });

            if (result.code === 0) {
                const categories = result.data.map(item => ({
                    ...item,
                    expanded: false,
                    children: []
                }));

                for (let category of categories) {
                    const subResult = await util.callCloudFunction('device', {
                        action: 'listCategories',
                        parentId: category._id
                    });

                    if (subResult.code === 0) {
                        category.children = subResult.data.map(child => ({
                            ...child,
                            expanded: false,
                            devices: []
                        }));

                        for (let child of category.children) {
                            const deviceResult = await util.callCloudFunction('device', {
                                action: 'listDevices',
                                categoryId: child._id,
                                status: ''
                            });

                            if (deviceResult.code === 0) {
                                child.devices = deviceResult.data.list;
                            }
                        }
                    }
                }

                this.setData({ categories });
            }
        } catch (error) {
            console.error('加载分类失败:', error);
            util.showError('加载分类失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    onCategoryTap(e) {
        const category = e.currentTarget.dataset.category;
        const categories = this.data.categories.map(item => {
            if (item._id === category._id) {
                return { ...item, expanded: !item.expanded };
            }
            return item;
        });
        this.setData({ categories });
    },

    onSubCategoryTap(e) {
        const category = e.currentTarget.dataset.category;
        const categories = this.data.categories.map(item => ({
            ...item,
            children: item.children.map(child => {
                if (child._id === category._id) {
                    return { ...child, expanded: !child.expanded };
                }
                return child;
            })
        }));
        this.setData({ categories });
    },

    onAddCategory() {
        wx.showModal({
            title: '新增分类',
            editable: true,
            placeholderText: '请输入分类名称',
            success: async (res) => {
                if (res.confirm && res.content) {
                    const result = await util.callCloudFunction('device', {
                        action: 'createCategory',
                        name: res.content,
                        level: 1,
                        sort: 0
                    });

                    if (result.code === 0) {
                        util.showSuccess('创建成功');
                        this.loadCategories();
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    },

    onEditCategory(e) {
        const category = e.currentTarget.dataset.category;
        wx.showModal({
            title: '编辑分类',
            editable: true,
            placeholderText: '请输入分类名称',
            content: category.name,
            success: async (res) => {
                if (res.confirm && res.content) {
                    const result = await util.callCloudFunction('device', {
                        action: 'updateCategory',
                        categoryId: category._id,
                        name: res.content
                    });

                    if (result.code === 0) {
                        util.showSuccess('更新成功');
                        this.loadCategories();
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    },

    onAddDevice() {
        wx.navigateTo({
            url: '/pages/admin/device/edit'
        });
    },

    onEditDevice(e) {
        const device = e.currentTarget.dataset.device;
        wx.navigateTo({
            url: `/pages/admin/device/edit?id=${device._id}&name=${device.name}&categoryId=${device.categoryId}&remark=${device.remark || ''}`
        });
    },

    onUpdateQuantity(e) {
        const device = e.currentTarget.dataset.device;
        wx.showModal({
            title: '修改数量',
            editable: true,
            placeholderText: '请输入新的总数量',
            content: device.totalQuantity.toString(),
            success: async (res) => {
                if (res.confirm && res.content) {
                    const quantity = parseInt(res.content);
                    if (isNaN(quantity) || quantity < 0) {
                        util.showError('请输入有效的数量');
                        return;
                    }

                    const result = await util.callCloudFunction('device', {
                        action: 'updateDeviceQuantity',
                        deviceId: device._id,
                        totalQuantity: quantity
                    });

                    if (result.code === 0) {
                        util.showSuccess('更新成功');
                        this.loadCategories();
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    },

    onToggleDeviceStatus(e) {
        const device = e.currentTarget.dataset.device;
        const action = device.status === 'active' ? 'disableDevice' : 'enableDevice';
        const text = device.status === 'active' ? '禁用' : '启用';

        wx.showModal({
            title: '提示',
            content: `确定要${text}该设备吗？`,
            success: async (res) => {
                if (res.confirm) {
                    const result = await util.callCloudFunction('device', {
                        action,
                        deviceId: device._id
                    });

                    if (result.code === 0) {
                        util.showSuccess(`${text}成功`);
                        this.loadCategories();
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    }
});
