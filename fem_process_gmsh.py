# %%
import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any

# %% [markdown]
# 1. .msh 파일을 일종의 text 파일처럼 import / 이게 불가하면 먼저 import하고 text 데이터로 변환 후 변수로 data_msh라는 변수로 저장
# 2. dict_section이라는 이름의 빈 딕셔너리 하나 만들자.
# 3. 이 data_msh 변수를 대상으로, 첫 '\$'가 나오는 줄에서 '\$' 다음에 나오는 텍스트를 name_section라는 이름의 변수로 저장
# 4. 그 다음 줄부터를 탐색 대상으로 하여 '\$' + name_section 이라는 문자열이 나오는 줄을 찾고, 첫 '\$'이 나왔던 줄의 다음 줄부터 해당 줄 바로 앞줄까지의 데이터를 복사하여 data_section이라는 변수로 저장
# 5. 그리고 방금 찾은 첫 '\$'가 나오는 줄에서부터 '\$' + name_section 이라는 문자열이 나오는 줄까지를 data_msh 변수에서 삭제
# 6. dict_section에 key = name_section, value = data_section 으로 하여 저장
# 7. 다시 data_msh 변수를 대상으로 더 이상 data_msh에 텍스트가 남아있지 않을 때까지 3~5 반복
# 8. dict_section에서 key = PhysicalNames에 해당하는 value를 대상으로 "BC"라는 텍스트가 들어있는 줄을 추출하여 하나의 변수(info_BC) 안에 저장
# 9. 8에서 추출된 BC가 적용될 PhysicalNames에 대해, 이게 몇 차원의 entity에 속한 건지, Tag는 몇인지 읽고, 해당 차원의 entity 데이터(dict_section의 Entities key에 해당하는 value)로 가서 각 entity들마다의 physicalTag를 읽으면서 해당 PhysicalNames Tag에 대응되는 entity의 Tag 확인하고 이걸 info_BC에서 대응되는 각 줄의 맨 끝에 추가
# 10. dict_section에서 key = 'Nodes'에 해당하는 value를 info_nodes라는 변수로 생성
# 11. dict_section에서 key = 'Elements'에 해당하는 value를 info_elements라는 변수로 생성
# 12. info_elements에서 info_BC의 각 BC의 entity 차원 및 Tag에 대응되는 element들 찾아서 걔네들의 node Tag만 따로 뺀 다음 중복 Tag는 하나만 담게 하여(unique filtering) info_BC의 대응되는 BC 줄의 우측에 추가
# 13. info_nodes라는 변수에서 nodes의 index, 글로벌 좌표만 나래비로 뽑아서 data_nodes라는 변수로 지정
# 14. info_elements라는 변수에서 elements의 type, index, 가지고 있는 node의 index만 나래비로 뽑아서 data_elements라는 변수로 지정
# 
# 이 모든 걸 구현하는 코드 짜줘.

# %%
# ----------------------------
# 데이터 구조
# ----------------------------
@dataclass
class BCInfo:
    name: str                 # "displacementBC" / "loadBC"
    dim: int                  # physical dimension (e.g., 2 for surface)
    physical_tag: int         # physical group tag
    entity_tag: int | None = None  # 해당 physical tag가 붙은 geometric entity tag (예: surface tag)
    node_tags: List[int] | None = None  # 해당 BC entity에 속한 element들의 node tag (unique)

# ----------------------------
# 1~7: 섹션 분해 로직 (요구사항대로 data_msh에서 잘라내며 dict_section에 저장)
# ----------------------------
def split_msh_sections_to_dict(path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    1) 파일을 텍스트로 읽어서 data_msh(list[str])로 저장
    2) dict_section 초기화
    3~7) data_msh에서 $Section ... $EndSection 구간을 잘라 dict_section[name]=data_section으로 저장
         그리고 해당 구간은 data_msh에서 삭제
    """
    
    # 1) 텍스트 파일 불러와서 줄 단위로 분리
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data_msh = f.read().splitlines() # 파일을 텍스트 데이터로 불러와 줄(줄바꿈) 단위로 분리하여 리스트에 저장

    # 2) dict_section 초기화
    dict_section: Dict[str, List[str]] = {}

    # 보조 함수: data_msh에서 첫 번째 $로 시작하는 라인 인덱스 찾기
    def find_first_section_start(lines: List[str]) -> int | None:
        for idx, line in enumerate(lines):
            s = line.strip()
            # 섹션 시작은 "$"로 시작하고 "$End"로 시작하지 않는 라인
            if s.startswith("$") and not s.startswith("$End"):
                return idx
        print('No more sections found.\n')
        return None

    # 3~7) data_msh에서 $Section ... $EndSection 구간을 잘라 dict_section[name]=data_section으로 저장, 그리고 해당 구간은 data_msh에서 삭제하는 과정을 반복(더이상 $로 시작하는 라인이 없을 때까지)
    while True:
        i0 = find_first_section_start(data_msh) # 첫 번째 $로 시작하는 라인 인덱스
        if i0 is None:
            break

        # 3) '$' 다음 텍스트를 name_section으로
        name_section = data_msh[i0].strip()[1:].strip()  # "$Nodes" -> "Nodes"
        end_marker = "$End" + name_section

        # 4) 다음 줄부터 end_marker 찾기
        i1 = None
        for j in range(i0 + 1, len(data_msh)):
            if data_msh[j].strip() == end_marker:
                i1 = j
                break
        if i1 is None:
            raise ValueError(f"섹션 {name_section}의 종료 마커({end_marker})를 찾지 못했습니다.")

        data_section = data_msh[i0 + 1 : i1]  # 시작 다음 줄부터 종료 바로 전까지

        # 5) data_msh에서 섹션 전체 삭제 (시작 줄 ~ 종료 줄)
        del data_msh[i0 : i1 + 1]

        # 6) dict_section 저장
        dict_section[name_section] = data_section

    return data_msh, dict_section


# ----------------------------
# 8: PhysicalNames에서 BC 라인만 필터링해서 해당 BC(PhysicalGroup)에 대응되는 이름, geometric entity의 차원, physicalTag 추출
# ----------------------------
def parse_physicalnames_for_bcs(physical_lines: List[str], keyword: str = "BC") -> List[BCInfo]:
    """
    msh4 
      $PhysicalNames:
      <numPhysicalNames>
      <dim> <physicalTag> "name"
      ...
      
      ※ $PhysicalNames 섹션 : PhysicalGroup의 정보를 담고 있는 섹션 ; PhysicalGroup: 노드/곡선/표면/체적 등 기하학적 엔티티에 임의의 태그와 이름을 붙여 그룹화한 것
      ※ physicalTag: 해당 PhysicalGroup의 Tag ; $Entities 섹션에서 해당 PhysicalGroup과 매핑되는 entityTag를 찾는 데 사용됨
    """
    if not physical_lines:
        raise ValueError("$PhysicalNames 섹션이 비어 있습니다.")

    n = int(physical_lines[0].strip())
    list_bc: List[BCInfo] = []

    for line in physical_lines[1:1+n]:
        s = line.strip()
        # 예: 2 106 "DirichletBC"
        parts = s.split()
        if len(parts) < 3: # BC가 아닌 라인 무시
            continue

        dim = int(parts[0])
        physical_tag = int(parts[1])

        # name은 큰따옴표 포함. 공백 있을 수도 있으니 join 후 처리
        rest = " ".join(parts[2:])
        if '"' in rest:
            name = rest.strip()
            if name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            else:
                name = name.replace('"', "")
        else:
            name = rest

        # BC가 들어있는 라인만 필터링해서 out 리스트에 추가
        if keyword in name:
            list_bc.append(BCInfo(name=name, dim=dim, physical_tag=physical_tag))

    # BC가 하나도 없을 경우 경고
    if not list_bc:
        print(f"[경고] PhysicalNames에서 '{keyword}'를 포함한 BC를 찾지 못했습니다.")

    return list_bc # [name, dim, physical_tag]

# ----------------------------
# 9: Entities에서 physicalTag 있는 geometric 엔티티에 한하여 entityTag <-> physicalTag 매핑 찾기
# ----------------------------
def parse_entities_physical_map(entities_lines: List[str]) -> Dict[Tuple[int, int], Set[int]]:
    """
    msh4
      $Entities:
      <numPoints> <numCurves> <numSurfaces> <numVolumes>
      (point blocks) : pointTag x y z numPhysicalTags [physicalTags]
      (curve blocks) : curveTag bbox(6) numPhysicalTags [physicalTags] numBoundingPoints pointTags(바운더리를 이루는 point tags)
      (surface blocks) : surfaceTag bbox(6) numPhysicalTags [physicalTags] numBoundingCurves curveTags(바운더리를 이루는 curve tags)
      (volume blocks) : volumeTag bbox(6) numPhysicalTags [physicalTags] numBoundingSurfaces surfaceTags(바운더리를 이루는 surface tags)
      
      ※ $Entities 섹션 : 기하학적 엔티티(점, 곡선, 표면, 체적)들의 정보를 담고 있는 섹션
      ※ bbox: 해당 기하학적 엔티티가 딱 맞게 들어가는 직육면체의 x, y, z 바운더리 (xmin, ymin, zmin, xmax, ymax, zmax)
      ※ physicalTags: 해당 기하학적 엔티티에 대응되는 PhysicalGroup의 Tag(없을 수도 있고 여러 개일 수도 있음)

    geometry 엔티티 중 대응되는 physicalTag가 있는 엔티티만 선별해서 (dim, entityTag) -> set(physicalTags) 매핑을 반환
    반환 형태 : {(dim, entityTag) : set(physicalTags)} ; 딕셔너리 키는 (dim, entityTag) 튜플, 값은 해당 엔티티에 붙은 physicalTags의 집합
    """
    if not entities_lines:
        raise ValueError("$Entities 섹션이 비어 있습니다.")

    header = entities_lines[0].split()
    if len(header) != 4:
        raise ValueError("$Entities 첫 줄 형식이 예상과 다릅니다.")
    npnt, ncrv, nsrf, nvol = map(int, header)

    idx = 1
    ent_phys: Dict[Tuple[int, int], Set[int]] = {}

    # ---- Points: pointTag x y z numPhysicalTags [physicalTags]
    for _ in range(npnt):
        parts = entities_lines[idx].split()
        idx += 1
        point_tag = int(parts[0])
        num_phys = int(parts[4])
        phys_tags = set(map(int, parts[5:5+num_phys])) if num_phys > 0 else set()
        ent_phys[(0, point_tag)] = phys_tags

    # ---- Curves: curveTag bbox(6) numPhysicalTags [physicalTags] numBoundingPoints pointTags
    for _ in range(ncrv):
        parts = entities_lines[idx].split()
        idx += 1
        curve_tag = int(parts[0])
        num_phys = int(parts[7])
        p0 = 8 # physicalTags 시작 인덱스
        phys_tags = set(map(int, parts[p0:p0+num_phys])) if num_phys > 0 else set()
        p1 = p0 + num_phys # numBoundingPoints 인덱스
        num_bpts = int(parts[p1])
        # bounding points는 p1+1 ~ p1+num_bpts (부호는 orientation)
        ent_phys[(1, curve_tag)] = phys_tags
        # (여기서는 bounding points 자체는 필요 없어서 저장 안 함)

    # ---- Surfaces: surfaceTag bbox(6) numPhysicalTags [physicalTags] numBoundingCurves curveTags
    for _ in range(nsrf):
        parts = entities_lines[idx].split()
        idx += 1
        srf_tag = int(parts[0])
        num_phys = int(parts[7])
        p0 = 8
        phys_tags = set(map(int, parts[p0:p0+num_phys])) if num_phys > 0 else set()
        p1 = p0 + num_phys
        num_bcrv = int(parts[p1])
        # bounding curves는 p1+1 ~ p1+num_bcrv (부호는 orientation)
        ent_phys[(2, srf_tag)] = phys_tags

    # ---- Volumes: volumeTag bbox(6) numPhysicalTags [physicalTags] numBoundingSurfaces surfaceTags
    for _ in range(nvol):
        parts = entities_lines[idx].split()
        idx += 1
        vol_tag = int(parts[0])
        num_phys = int(parts[7])
        p0 = 8
        phys_tags = set(map(int, parts[p0:p0+num_phys])) if num_phys > 0 else set()
        p1 = p0 + num_phys
        num_bsrf = int(parts[p1])
        ent_phys[(3, vol_tag)] = phys_tags

    return ent_phys

# 앞서 뽑은 entityTag <-> physicalTag 매핑을 이용하여, BCInfo 리스트의 각 BC(PhysicalGroup)에다가 대응되는 entity_tag 붙이기
def attach_entity_tags_to_bcs(bcs: List[BCInfo], ent_phys_map: Dict[Tuple[int, int], Set[int]]) -> None:
    """
    9) 각 BC (dim, physical_tag)에 대해, 그 physical_tag를 포함하는 (dim, entity_tag)를 찾아 bc.entity_tag에 저장.
    """
    # dim별로 후보를 좁히면 빠름
    by_dim: Dict[int, List[Tuple[int, Set[int]]]] = {0: [], 1: [], 2: [], 3: []}
    for (dim, ent_tag), phys_tags in ent_phys_map.items():
        by_dim[dim].append((ent_tag, phys_tags))

    for bc in bcs:
        cand = by_dim.get(bc.dim, [])
        hits = [ent_tag for (ent_tag, phys_tags) in cand if bc.physical_tag in phys_tags]
        if not hits:
            bc.entity_tag = None
            print(f"[경고] BC '{bc.name}'(dim={bc.dim}, phys={bc.physical_tag})에 대응되는 entity_tag를 $Entities에서 못 찾았습니다.")
        elif len(hits) == 1:
            bc.entity_tag = hits[0]
        else:
            # 보통 1개여야 정상인데, 여러 엔티티에 같은 physical tag를 걸 수도 있어서
            # 여기서는 첫 번째만 붙이고 경고
            bc.entity_tag = hits[0]
            print(f"[경고] BC '{bc.name}'에 entity_tag 후보가 여러 개입니다: {hits}. 첫 번째({hits[0]})만 사용합니다.")


# ----------------------------
# 10~12: Nodes/Elements 파싱 + BC 요소에서 노드 유니크 추출
# ----------------------------
def parse_nodes_msh4(info_nodes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    msh4 $Nodes:
      numEntityBlocks totalNumNodes minTag maxTag
      (block 반복)
        entityDim entityTag parametric numNodesInBlock
        nodeTags (numNodesInBlock개, 여러 줄에 걸칠 수 있음)
        coords   (numNodesInBlock줄, x y z [parametric...])
    return:
      node_tags (N,)
      xyz (N,3)
    """
    if not info_nodes:
        raise ValueError("$Nodes 섹션이 비어 있습니다.")

    # 첫 줄(전체 노드 헤더) 파싱 : 전체 geometry 블록 개수, 전체 노드 개수, 최소 태그, 최대 태그(텍스트 -> 정수 변환)
    num_blocks, total_nodes, min_tag, max_tag = map(int, info_nodes[0].split())
    # geometry 블록 단위 파싱 준비(각 geometry 블록 내 노드 tag, 좌표 담을 리스트 초기화)
    node_tags: List[int] = []
    xyz: List[Tuple[float, float, float]] = []

    idx = 1 # geometry 블록 단위 파싱 시 두 번째 줄부터 시작
    # geometry 블록 단위 파싱
    for _ in range(num_blocks):
        ent_dim, ent_tag, parametric, nblock = map(int, info_nodes[idx].split()) # 현재 블록 헤더 파싱 : 블록 차원, 블록 태그, 파라메트릭 여부, 블록 내 노드 개수(텍스트 -> 정수 변환)
        idx += 1 # 다음 줄로 이동

        # 현재 geometry 블록 내 노드 태그 (packed) 저장
        tags = []
        while len(tags) < nblock: # 현재 블록 내 노드 개수만큼 태그 수집
            tags.extend(map(int, info_nodes[idx].split())) # 한 줄에 여러 개 있을 수 있으니 split 후 모두 추가
            idx += 1 # 다음 줄로 이동

        # geometry 블록 내 노드 좌표(xyz) 저장
        for k in range(nblock): # 현재 블록 내 노드 개수만큼 좌표 수집
            parts = info_nodes[idx].split()
            idx += 1
            x, y, z = map(float, parts[:3])
            node_tags.append(tags[k])
            xyz.append((x, y, z))
            # parametric 좌표는 무시

    # node_tags와 xyz를 numpy 배열로 변환
    node_tags_arr = np.array(node_tags, dtype=int)
    xyz_arr = np.array(xyz, dtype=float)

    # node_tag 기준 정렬
    order = np.argsort(node_tags_arr)
    node_tags_arr = node_tags_arr[order]
    xyz_arr = xyz_arr[order]

    return node_tags_arr, xyz_arr # (N,), (N,3) ndarray 반환

def parse_elements_msh4(info_elements: List[str]) -> List[Tuple[int, int, List[int], int, int]]:
    """
    msh4 $Elements:
      numEntityBlocks totalNumElements minTag maxTag
      block 반복:
        entityDim entityTag elementType numElementsInBlock
        elemTag node1 node2 ...
    return list of tuples:
      (elem_tag, elem_type, conn_node_tags, entity_dim, entity_tag)
    """
    if not info_elements:
        raise ValueError("$Elements 섹션이 비어 있습니다.")

    # 첫 줄(전체 요소 헤더) 파싱 : 전체 geometry 블록 개수, 전체 요소 개수, 최소 태그, 최대 태그(텍스트 -> 정수 변환)
    num_blocks, total_elems, min_tag, max_tag = map(int, info_elements[0].split())
    
    # geometry 블록 단위 파싱 준비
    elements: List[Tuple[int, int, List[int], int, int]] = [] # (elem_tag, elem_type, conn_node_tags, entity_dim, entity_tag) 형태로 저장
    idx = 1 # geometry 블록 단위 파싱 시 두 번째 줄부터 시작

    # geometry 블록 단위 파싱
    for _ in range(num_blocks):
        ent_dim, ent_tag, elem_type, nelements = map(int, info_elements[idx].split()) # 현재 geometry 블록 헤더 파싱 : geometry 블록 차원, geometry 블록 태그, 요소 타입, geometry 블록 내 요소 개수(텍스트 -> 정수 변환)
        idx += 1
        for _k in range(nelements): # 현재 geometry 블록 내 요소 개수만큼 요소 데이터 수집
            parts = list(map(int, info_elements[idx].split()))
            idx += 1
            elem_tag = parts[0]
            conn = parts[1:]
            elements.append((elem_tag, elem_type, conn, ent_dim, ent_tag))

    # elem_tag 정렬
    elements.sort(key=lambda t: t[0])
    return elements # (elem_tag, elem_type, conn_node_tags, entity_dim, entity_tag) 리스트 반환


def attach_bc_node_tags_from_elements(bcs: List[BCInfo], elements: List[Tuple[int, int, List[int], int, int]]) -> None:
    """
    12) data_elements에서 BC의 (dim, entity_tag)에 해당하는 element들을 찾고,
        그 element들의 node tag를 유니크하게 모아서 bc.node_tags에 저장
    """
    # (ent_dim, ent_tag) -> set(node_tags)
    bucket: Dict[Tuple[int, int], Set[int]] = {}

    for (elem_tag, elem_type, conn, ent_dim, ent_tag) in elements:
        key = (ent_dim, ent_tag)
        if key not in bucket:
            bucket[key] = set()
        bucket[key].update(conn)

    for bc in bcs:
        if bc.entity_tag is None:
            bc.node_tags = []
            continue
        key = (bc.dim, bc.entity_tag)
        nodes = sorted(bucket.get(key, set()))
        bc.node_tags = nodes


# ----------------------------
# 13~14: data_nodes / data_elements 생성
# ----------------------------
def build_data_nodes(node_tags: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    13) node tag / x / y / z 형태로 (N,4) 행렬 생성
    """
    return np.column_stack([node_tags.astype(float), xyz])


def build_data_elements(elements: List[Tuple[int, int, List[int], int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    14) element tag / element type / connectivity 를 '행렬 데이터'로 구성
    - elem_meta: (E,2) = [elem_tag, elem_type]
    - conn_padded: (E,max_nodes) connectivity를 -1 padding
    """
    E = len(elements)
    elem_tags = np.array([e[0] for e in elements], dtype=int)
    elem_types = np.array([e[1] for e in elements], dtype=int)
    elem_meta = np.column_stack([elem_tags, elem_types])

    max_nodes = max((len(e[2]) for e in elements), default=0) # 요소 한 개당 최대 노드 수 추출
    conn_padded = -np.ones((E, max_nodes), dtype=int) # -1로 패딩된 연결성(요소가 가진 노드 태그) 행렬 초기화
    for i, (_etag, _etype, conn, _d, _t) in enumerate(elements):
        conn_padded[i, :len(conn)] = conn # 실제 연결성 데이터로 채우기
    # 이렇게 함으로써 각 요소가 가진 노드 수가 달라도 동일한 shape의 행렬로 표현 가능(다른 요소에 비해 노드 수가 적은 요소의 해당 노드 태그 부분은 -1로 채움)
    data_elements = np.hstack((elem_meta, conn_padded))
    # 14) 결과 반환
    return data_elements


# ----------------------------
# 전체 파이프라인 (네 1~14를 실행)
# ----------------------------
def run_pipeline(path_msh: str) -> Dict[str, Any]:
    # 1~7 : 원본 msh 파일에서 섹션 개별 추출 후 딕셔너리에 저장
    remaining, dict_section = split_msh_sections_to_dict(path_msh) # remaining: 원본에서 섹션 다 도려내고 남은 텍스트 라인, dict_section: 섹션별 데이터 딕셔너리

    # 8 : PhysicalNames 섹션에서 Boundary Conditions 라인만 추출
    bcs = parse_physicalnames_for_bcs(
        dict_section.get("PhysicalNames", []),
        keyword="BC",
    )

    # 9 : Entities 섹션에서 physicalTag -> entityTag 매핑 추출 후, bcs에 entityTag 붙이기
    ent_phys_map = parse_entities_physical_map(dict_section.get("Entities", []))
    attach_entity_tags_to_bcs(bcs, ent_phys_map)

    # 10~11 : Nodes/Elements 섹션 파싱
    info_nodes = dict_section.get("Nodes", []) #  Nodes 섹션 추출 ; Nodes key가 있으면 value(라인 리스트) 추출, 없으면 빈 리스트 반환
    info_elements = dict_section.get("Elements", []) # Elements 섹션 추출 ; Elements key가 있으면 value(라인 리스트) 추출, 없으면 빈 리스트 반환

    node_tags, xyz = parse_nodes_msh4(info_nodes) # 추출한 Nodes 섹션에서 노드별 태그 및 좌표 정보만 파싱
    elements = parse_elements_msh4(info_elements) # 추출한 Elements 섹션에서 요소별 태그, 노드 connectivity, geometry 엔티티 정보만 파싱

    # 12 : BC별로 대응되는 노드 태그 유니크하게 추출하여 bcs에 붙이기
    attach_bc_node_tags_from_elements(bcs, elements)

    # 13 : data_nodes 생성
    data_nodes = build_data_nodes(node_tags, xyz)
    df_data_nodes = pd.DataFrame(data_nodes)
    df_data_nodes.columns = ['node_tag', 'x', 'y', 'z']  # node_tag, x, y, z

    # 14
    data_elements = build_data_elements(elements)
    df_data_elements = pd.DataFrame(data_elements)
    df_data_elements.columns = ['elem_tag', 'elem_type_int'] + [f'node_{i+1}' for i in range(data_elements.shape[1]-2)]
    type_map = {
                # 1D elements
                1:  "line2",        # 2-node line
                8:  "line3",        # 3-node quadratic line

                # 2D elements
                2:  "tri3",         # 3-node triangle
                3:  "quad4",        # 4-node quadrilateral
                9:  "tri6",         # 6-node quadratic triangle
                10: "quad9",        # 9-node quadratic quadrilateral
                16: "quad8",        # 8-node serendipity quad

                # 3D elements
                4:  "tetra4",       # 4-node tetrahedron
                11: "tetra10",      # 10-node quadratic tetrahedron

                5:  "hexa8",        # 8-node hexahedron
                12: "hexa27",       # 27-node quadratic hexahedron
                17: "hexa20",       # 20-node serendipity hexahedron

                6:  "prism6",       # 6-node wedge (prism)
                13: "prism18",      # 18-node quadratic prism
                18: "prism15",      # 15-node serendipity prism

                7:  "pyramid5",     # 5-node pyramid
                14: "pyramid14"}   # 14-node quadratic pyramid
    
    df_data_elements.loc[:, 'elem_type'] = df_data_elements['elem_type_int'].map(type_map).fillna(df_data_elements['elem_type_int'])

    # 결과 정리
    list_info_BC = [
        {
            "name": bc.name,
            "dim": bc.dim,
            "physical_tag": bc.physical_tag,
            "entity_tag": bc.entity_tag,
            "node_tags": bc.node_tags,
            }
        for bc in bcs
    ]

    print("data_nodes shape:", data_nodes.shape)
    print("data_elements shape:", data_elements.shape)
    print()
    print("BC summary:")
    for info_BC in list_info_BC:
        print(info_BC["name"]+':', "dim=", info_BC["dim"], "physical_tag=", info_BC["physical_tag"],
              "entity=", info_BC["entity_tag"], "num_nodes=", len(info_BC["node_tags"] or []))

    dict_data_processed = {
        "dict_section": dict_section,
        "info_BC": list_info_BC,
        "data_nodes": df_data_nodes,                 # (N,4): [node_tag, x, y, z]
        "data_elements": df_data_elements,     # (E,2+max_nodes): [elem_tag, elem_type] + connectivity padded with -1
        "remaining_text_lines": remaining,         # 섹션 다 뽑고 남은 텍스트(보통 빈 리스트)
    }

    return dict_data_processed

# ----------------------------
# 실행 예시
# ----------------------------
if __name__ == "__main__":
    path = "C:\\Users\\opew4\\Desktop\\Files_FreeCAD\\square_hex.msh"  # 네 파일 경로
    out = run_pipeline(path)