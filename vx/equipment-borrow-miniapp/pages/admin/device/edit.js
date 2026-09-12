const auth = require('../../../../utils/auth');
const util = require('../../../../utils/util');

Page({
    data: {
        isEdit: false,
        deviceId: '',
        name: '',
        categoryId: '',
        remark: '',
        categories: [],
        categoryIndex: 0,
        loading: false
    },

    async onLoad(options) {
        if (options.id) {
            this.setData({
                isEdit: true,
                deviceId: options.id,
                name: decodeURIComponent(options.name || ''),
                categoryId: options.categoryId || '',
                remark: decodeURIComponent(options.remark || '')
            });
            wx.setNavigationBarTitle({ title: '编辑设备' });
        } else {
            wx.setNavigationBarTitle({ title: '新增设备' });
        }
        await this.loadCategories();

        if (options.categoryId) {
            const idx = this.data.categories.findIndex(c => c._id === options.categoryId);
            if (idx >= 0) this.setData({ categoryIndex: idx });
        }
    },

    async loadCategories() {
        try {
            const result = await util.callCloudFunction('device', {
                action: 'listAllCategories'
            });
            if (result.code === 0) {
                const level3 = result.data.filter(c => c.level === 3);
                const catMap = {};
                result.data.forEach(c => catMap[c._id] = c);

                const categories = level3.map(c => {
                    const parent = catMap[c.parentId];
                    const grandParent = parent ? catMap[parent.parentId] : null;
                    let path = c.name;
                    if (parent) path = `${parent.name} > ${path}`;
                    if (grandParent) path = `${grandParent.name} > ${path}`;
                    return { _id: c._id, name: path };
                });

                this.setData({ categories });

                if (this.data.categoryId) {
                    const idx = categories.findIndex(c => c._id === this.data.categoryId);
                    if (idx >= 0) {
                        this.setData({ categoryIndex: idx });
                    } else {
                        const cat = catMap[this.data.categoryId];
                        if (cat) {
                            let path = cat.name;
                            const parent = catMap[cat.parentId];
                            const grandParent = parent ? catMap[parent.parentId] : null;
                            if (parent) path = `${parent.name} > ${path}`;
                            if (grandParent) path = `${grandParent.name} > ${path}`;
                            categories.push({ _id: cat._id, name: path });
                            this.setData({ categories, categoryIndex: categories.length - 1 });
                        }
                    }
                }
            }
        } catch (error) {
            util.showError('加载分类失败');
        }
    },

    onNameInput(e) { this.setData({ name: e.detail.value }); },
    onRemarkInput(e) { this.setData({ remark: e.detail.value }); },

    onCategoryChange(e) {
        this.setData({ categoryIndex: e.detail.value });
    },

    async handleSubmit() {
        const { isEdit, deviceId, name, categoryId, remark, categories, categoryIndex } = this.data;

        if (!name.trim()) { util.showError('请输入设备名称'); return; }
        if (categories.length === 0) { util.showError('请先创建三级分类'); return; }

        const selectedCategoryId = categories[categoryIndex]?._id;
        if (!selectedCategoryId) { util.showError('请选择分类'); return; }

        this.setData({ loading: true });

        try {
            const action = isEdit ? 'updateDevice' : 'createDevice';
            const data = { action, name, categoryId: selectedCategoryId, remark };
            if (isEdit) data.deviceId = deviceId;

            const result = await util.callCloudFunction('device', data);

            if (result.code === 0) {
                util.showSuccess(isEdit ? '修改成功' : '创建成功');
                setTimeout(() => wx.navigateBack(), 1500);
            } else {
                util.showError(result.message);
            }
        } catch (error) {
            util.showError('操作失败');
        } finally {
            this.setData({ loading: false });
        }
    }
});
