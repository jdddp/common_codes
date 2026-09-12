const auth = require('../../utils/auth');
const util = require('../../utils/util');

Page({
    data: {
        tree: [],
        filteredTree: [],
        keyword: '',
        loading: false
    },

    onLoad() {
        this.loadData();
    },

    onShow() {
        this.loadData();
    },

    async loadData() {
        this.setData({ loading: true });
        try {
            const [catRes, devRes] = await Promise.all([
                util.callCloudFunction('device', { action: 'listAllCategories' }),
                util.callCloudFunction('device', { action: 'listDevices', status: 'active' })
            ]);
            if (catRes.code === 0 && devRes.code === 0) {
                const tree = this.buildTree(catRes.data, devRes.data.list);
                this.setData({ tree, filteredTree: tree });
            }
        } catch (error) {
            util.showError('加载失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    buildTree(categories, devices) {
        const catMap = {};
        const roots = [];
        const deviceList = [];

        categories.forEach(c => {
            catMap[c._id] = { ...c, children: [], devices: [], expanded: false };
        });

        categories.forEach(c => {
            if (c.parentId && catMap[c.parentId]) {
                catMap[c.parentId].children.push(catMap[c._id]);
            } else {
                roots.push(catMap[c._id]);
            }
        });

        devices.forEach(d => {
            if (catMap[d.categoryId]) {
                catMap[d.categoryId].devices.push(d);
                d.fullPath = this.getFullPath(catMap, d.categoryId);
            }
            deviceList.push(d);
        });

        return roots;
    },

    getFullPath(catMap, categoryId) {
        const parts = [];
        let current = catMap[categoryId];
        while (current) {
            parts.unshift(current.name);
            current = current.parentId ? catMap[current.parentId] : null;
        }
        return parts.join(' > ');
    },

    onSearchInput(e) {
        this.setData({ keyword: e.detail.value });
    },

    onSearch() {
        const { keyword, tree } = this.data;
        if (!keyword.trim()) {
            this.setData({ filteredTree: tree });
            return;
        }

        const filtered = this.filterTree(tree, keyword.toLowerCase());
        this.setData({ filteredTree: filtered });
    },

    filterTree(nodes, keyword) {
        return nodes.map(node => {
            const matchedDevices = node.devices.filter(d =>
                d.name.toLowerCase().includes(keyword) ||
                (d.fullPath && d.fullPath.toLowerCase().includes(keyword))
            );

            const filteredChildren = this.filterTree(node.children || [], keyword);

            const nameMatch = node.name.toLowerCase().includes(keyword);
            const hasMatchedDevices = matchedDevices.length > 0;
            const hasMatchedChildren = filteredChildren.length > 0;

            if (nameMatch || hasMatchedDevices || hasMatchedChildren) {
                return {
                    ...node,
                    devices: matchedDevices,
                    children: filteredChildren,
                    expanded: true
                };
            }
            return null;
        }).filter(Boolean);
    },

    onToggleNode(e) {
        const id = e.currentTarget.dataset.id;
        this.toggleNode(this.data.filteredTree, id);
        this.setData({ filteredTree: this.data.filteredTree });
    },

    toggleNode(nodes, id) {
        for (let node of nodes) {
            if (node._id === id) { node.expanded = !node.expanded; return true; }
            if (node.children && this.toggleNode(node.children, id)) return true;
        }
        return false;
    },

    onBorrow(e) {
        const device = e.currentTarget.dataset.device;
        if (device.availableQuantity <= 0) {
            util.showError('暂无可用设备');
            return;
        }
        wx.navigateTo({
            url: `/pages/borrow/borrow?deviceId=${device._id}&deviceName=${encodeURIComponent(device.fullPath || device.name)}&availableQuantity=${device.availableQuantity}`
        });
    }
});
